import os
import threading

from agent.runners.base import Runner

try:  # pragma: no cover - only importable on a provisioned instance
    import soundfile as sf
    from huggingface_hub import hf_hub_download
except ImportError:  # tests patch these names directly
    sf = None
    hf_hub_download = None


RVC_DIR = "/opt/rvc"
WEIGHT_DIR = f"{RVC_DIR}/assets/weights"
INDEX_DIR = f"{RVC_DIR}/assets/indices"

# Clearly the best of RVC's pitch extractors on sung vocals; crepe and
# harvest are materially worse, pm worse still. Pitch tracking is what
# decides whether the converted vocal still holds the original melody.
DEFAULT_F0_METHOD = "rmvpe"


def load_vc(model_file: str):
    """Load RVC's own VC pipeline with a model selected.

    Imported inside the function, and after chdir, because RVC resolves its
    assets (hubert, rmvpe, weights) relative to the working directory and
    reads several of them at import time.

    There is no CLI to drive: an earlier version of this runner shelled out
    to `tools/infer_cli.py`, which does not exist in the repo (job 48 died
    on it). infer.vc.modules.VC is the supported headless entrypoint.
    """
    os.environ.setdefault("weight_root", WEIGHT_DIR)
    os.environ.setdefault("index_root", INDEX_DIR)
    os.environ.setdefault("rmvpe_root", f"{RVC_DIR}/assets/rmvpe")
    os.chdir(RVC_DIR)

    from configs.config import Config
    from infer.vc.modules import VC

    vc = VC(Config())
    vc.get_vc(model_file)
    return vc


class RVCRunner(Runner):
    """Retrieval-based Voice Conversion: re-sings an existing vocal in a
    target voice, keeping the original melody, timing and phrasing and
    changing only the timbre.

    This is the vocal half of a cover. The instrumental is left untouched,
    which is exactly why RVC covers sound like real records - the backing
    track never leaves the original mix.

    Feed it an isolated vocal (see the demucs runner), not a full mix:
    everything else in the input gets mangled by the conversion.
    """

    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _fetch(self, params: dict) -> tuple[str | None, str]:
        """The voice model, and its retrieval index if one was given."""
        model_file = params.get("model_file")
        if not model_file:
            return None, ""
        if params.get("model_repo"):
            hf_hub_download(params["model_repo"], model_file, local_dir=WEIGHT_DIR)
        index_path = params.get("index_path", "")
        if params.get("model_repo") and params.get("index_file"):
            index_path = hf_hub_download(
                params["model_repo"], params["index_file"], local_dir=INDEX_DIR
            )
        return model_file, index_path

    def _execute(self, input_path: str, params: dict) -> None:
        self._status = "running"
        try:
            model_file, index_path = self._fetch(params)
            if not model_file:
                self._log(
                    "no voice model given - set params.model_file, plus "
                    "model_repo to pull it from HuggingFace"
                )
                self._status = "failed"
                self._output_path = None
                return

            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "converted.wav")

            vc = load_vc(model_file)
            status, (rate, audio) = vc.vc_single(
                0,                                  # sid: speaker in a multi-speaker model
                input_path,
                # Semitones. 0 by default: transposing changes the melody,
                # which is the one thing a cover must not do.
                int(params.get("transpose", 0)),
                params.get("f0method", DEFAULT_F0_METHOD),
                index_path,
                # How hard the retrieval index pulls timbre toward the
                # training data: higher matches the target voice more closely
                # but drags in more of its artifacts.
                float(params.get("index_rate", 0.66)),
                int(params.get("resample_sr", 0)),   # 0 keeps RVC's own rate
                float(params.get("rms_mix_rate", 0.25)),
                # Shields voiceless consonants and breaths from conversion;
                # lower protects more.
                float(params.get("protect", 0.33)),
            )

            if audio is None:
                self._log(f"conversion returned no audio: {status}")
                self._status = "failed"
                self._output_path = None
                return

            sf.write(output_path, audio, rate)
            if os.path.exists(output_path):
                self._output_path = output_path
                self._status = "done"
            else:
                self._log(f"RVC reported success but wrote nothing to {output_path}")
                self._status = "failed"
                self._output_path = None
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
