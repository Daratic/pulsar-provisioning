import os
import subprocess
import threading
import urllib.request

from agent.runners.base import Runner

# Magenta's pretrained timbre-transfer checkpoints (Apache-2.0), plain HTTPS.
# Only step 20000 exists in this bucket, and the transfer script restores it
# by name.
CHECKPOINT_BASE = "https://storage.googleapis.com/ddsp/models/timbre_transfer_colab/2021-07-08"
CHECKPOINT_FILES = (
    "operative_config-0.gin",
    "dataset_statistics.pkl",
    "ckpt-20000.index",
    "ckpt-20000.data-00000-of-00001",
)
INSTRUMENTS = ("tenor_saxophone", "flute", "flute2", "trumpet", "violin")
# The bucket holds one checkpoint step per instrument, and it is not the same
# for all of them: violin is ckpt-40000 (job 60, 2026-10-16: 404 on 20000).
# Listed from the GCS bucket on 2026-10-16.
CHECKPOINT_STEPS = {"violin": 40000}
DEFAULT_CHECKPOINT_STEP = 20000


def checkpoint_step(instrument: str) -> int:
    return CHECKPOINT_STEPS.get(instrument, DEFAULT_CHECKPOINT_STEP)


def checkpoint_files(instrument: str) -> tuple[str, ...]:
    step = checkpoint_step(instrument)
    return (
        "operative_config-0.gin",
        "dataset_statistics.pkl",
        f"ckpt-{step}.index",
        f"ckpt-{step}.data-00000-of-00001",
    )
DEFAULT_INSTRUMENT = "tenor_saxophone"

# TensorFlow 2.11 has wheels for Python 3.7-3.10 only, so DDSP lives in its
# own venv, apart from the image's torch Python that runs the agent.
VENV_PYTHON = "/opt/ddsp/venv/bin/python"
TRANSFER_SCRIPT = "/opt/ddsp/transfer.py"
MODELS_ROOT = "/opt/ddsp/models"

_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


def checkpoint_url(instrument: str, filename: str) -> str:
    return f"{CHECKPOINT_BASE}/solo_{instrument}_ckpt/{filename}"


class _BadParams(ValueError):
    pass


def _parse_adjust(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise _BadParams(f"adjust must be on/off, got {value!r}")


class DdspTimbreRunner(Runner):
    """Timbre transfer with Magenta DDSP: a monophonic solo rendered as
    another instrument, tenor saxophone by default.

    Monophonic only - DDSP tracks ONE pitch (CREPE) and resynthesises it
    with a harmonic-plus-noise synth trained on the target instrument. Feed
    it a solo stem (a guitar lead, a vocal line), not a mix. For polyphonic
    material use the rave runner.

    The model runs in a separate Python 3.10 venv as a subprocess
    (/opt/ddsp/transfer.py), like demucs: TensorFlow 2.11 and the agent's
    torch Python cannot share an interpreter.
    """

    def __init__(self, workdir: str = "/workspace", models_root: str = MODELS_ROOT):
        self._workdir = workdir
        self._models_root = models_root
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        # Set synchronously, before the thread exists: a session reuses this
        # runner, and the previous job's "done" must not be readable by the
        # first poll after a new dispatch.
        self._status = "running"
        self._output_path = None
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _build_args(self, params: dict) -> list[str]:
        instrument = params.get("instrument") or DEFAULT_INSTRUMENT
        if instrument not in INSTRUMENTS:
            raise _BadParams(f"instrument must be one of {list(INSTRUMENTS)}, got {instrument!r}")

        octave_shift = params.get("octave_shift", "auto")
        if octave_shift is None or str(octave_shift).strip().lower() == "auto":
            octave_shift = "auto"
        else:
            try:
                octave_shift = str(int(str(octave_shift).strip()))
            except ValueError:
                raise _BadParams(f"octave_shift must be 'auto' or a whole number, got {octave_shift!r}")

        try:
            loudness = float(params.get("loudness_shift_db", 0.0) or 0.0)
        except (TypeError, ValueError):
            raise _BadParams(f"loudness_shift_db must be a number, got {params.get('loudness_shift_db')!r}")

        adjust = _parse_adjust(params.get("adjust", True))

        args = [
            "--instrument", instrument,
            "--models-root", self._models_root,
            "--octave-shift", octave_shift,
            "--loudness-shift-db", str(loudness),
            "--adjust", "1" if adjust else "0",
        ]
        rate = params.get("output_sample_rate")
        if rate not in (None, ""):
            try:
                rate = int(rate)
            except (TypeError, ValueError):
                raise _BadParams(f"output_sample_rate must be a whole number, got {rate!r}")
            if rate <= 0:
                raise _BadParams(f"output_sample_rate must be positive, got {rate}")
            args += ["--output-sample-rate", str(rate)]
        return args

    def _ensure_model(self, instrument: str) -> None:
        """Download an instrument's four files on first use. Tenor sax is
        fetched at install time; the others come here, into the same layout.
        Each file lands under a .part name first, so an interrupted download
        is never mistaken for a complete one."""
        model_dir = os.path.join(self._models_root, instrument)
        os.makedirs(model_dir, exist_ok=True)
        for filename in checkpoint_files(instrument):
            target = os.path.join(model_dir, filename)
            if os.path.exists(target):
                continue
            url = checkpoint_url(instrument, filename)
            self._log(f"downloading {url}")
            partial = target + ".part"
            try:
                urllib.request.urlretrieve(url, partial)
                os.replace(partial, target)
            finally:
                if os.path.exists(partial):
                    os.remove(partial)

    def _execute(self, input_path: str, params: dict) -> None:
        # Runs on a daemon thread: an escaping exception would leave status
        # pinned at "running" forever, so everything is caught.
        self._status = "running"
        self._output_path = None
        try:
            # A fresh log per job: the download lines and the script's output
            # below both append to it.
            open(self._log_path, "w").close()
            try:
                args = self._build_args(params)
            except _BadParams as exc:
                self._log(str(exc))
                self._status = "failed"
                return

            instrument = args[args.index("--instrument") + 1]
            self._ensure_model(instrument)

            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            expected = os.path.join(output_dir, f"ddsp_{instrument}.wav")
            if os.path.exists(expected):
                os.remove(expected)

            # The agent's PYTHONPATH points the torch Python at Pulsar; the
            # 3.10 venv must not pick it up (the notebook unsets it too).
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            cmd = [VENV_PYTHON, TRANSFER_SCRIPT, *args, "--in", input_path, "--out", expected]
            with open(self._log_path, "a") as log_file:
                process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
                process.wait()
            if process.returncode != 0:
                self._log(f"transfer.py exited {process.returncode}")
                self._status = "failed"
                return
            if os.path.exists(expected):
                self._output_path = expected
                self._status = "done"
            else:
                self._log(f"transfer.py exited 0 but wrote no output at {expected}")
                self._status = "failed"
        except Exception as exc:
            self._status = "failed"
            self._output_path = None
            self._log(f"_execute failed: {exc}")

    def _log(self, message: str) -> None:
        try:
            with open(self._log_path, "a") as log_file:
                log_file.write(f"\n[agent] {message}\n")
        except OSError:
            pass

    def status(self) -> str:
        return self._status

    def output_path(self) -> str | None:
        return self._output_path

    def log_tail(self, lines: int = 100) -> str:
        try:
            with open(self._log_path) as log_file:
                return "".join(log_file.readlines()[-lines:])
        except FileNotFoundError:
            return ""
