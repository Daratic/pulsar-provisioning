import os
import subprocess
import threading

from agent.runners.base import Runner

# htdemucs_ft, not the plain htdemucs default: it is the current best
# open-source vocal separator (9.19 dB median vocal SDR on MUSDB18-HQ). It
# costs roughly 4x the compute, which is immaterial on a rented 5090, and
# vocal quality is exactly what matters here - the stem is fed to RVC, and
# every artifact in it survives the voice conversion.
DEFAULT_DEMUCS_MODEL = "htdemucs_ft"

# The stem a cover pipeline wants. htdemucs models emit vocals/drums/bass/
# other; htdemucs_6s adds guitar and piano.
DEFAULT_STEM = "vocals"


class DemucsRunner(Runner):
    """Wraps the `demucs` CLI (stem separation). Pretrained weights are NOT
    baked into the image - Demucs downloads them to its torch hub cache on
    first invocation, per the startup-fetch decision in the design spec
    (section 2)."""

    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        # Set synchronously, before the thread exists: a session reuses this
        # runner, and the previous job's "done" and output must not be
        # readable by the first poll after a new dispatch.
        self._status = "running"
        self._output_path = None
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _execute(self, input_path: str, params: dict) -> None:
        # Runs on a daemon thread: an escaping exception would otherwise leave
        # status pinned at "idle" forever - see SeedVCRunner for the same fix.
        self._status = "running"
        try:
            model = params.get("model", DEFAULT_DEMUCS_MODEL)
            stem = params.get("stem", DEFAULT_STEM)
            with open(self._log_path, "w") as log_file:
                process = subprocess.Popen(
                    ["python3", "-m", "demucs", "-n", model, "-o", self._workdir, input_path],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                process.wait()
            if process.returncode != 0:
                self._status = "failed"
                self._output_path = None
                return

            name = os.path.splitext(os.path.basename(input_path))[0]
            # demucs writes to {-o}/{model}/{track}/{stem}.wav. The extra
            # "separated" component only appears when using its DEFAULT
            # output directory; we pass -o explicitly, so it must not be
            # here (job 36 separated the track correctly and then failed
            # looking in the wrong place).
            expected = os.path.join(self._workdir, model, name, f"{stem}.wav")
            # Checked rather than assumed: reporting "done" on the exit code
            # alone left output_path pointing at a file that might not exist,
            # and the problem only surfaced later as a 404 during retrieval,
            # by which point the instance was being torn down.
            if os.path.exists(expected):
                self._output_path = expected
                self._status = "done"
            else:
                self._log(f"demucs exited 0 but no stem at {expected}")
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
