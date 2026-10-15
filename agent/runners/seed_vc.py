import glob
import os
import subprocess
import threading

from agent.runners.base import Runner


class SeedVCRunner(Runner):
    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _execute(self, input_path: str, params: dict) -> None:
        # Runs on a daemon thread: any escaping exception would die silently and
        # leave status pinned at "idle", so the orchestrator would poll until its
        # 30-minute timeout. Catch everything and fail fast instead.
        self._status = "running"
        try:
            reference_path = params["reference_path"]
            # seed-vc's inference.py (real upstream script, not an installed
            # module) takes --output as a DIRECTORY, not a specific file path.
            # It names the resulting file itself (see comment below), so we
            # create the directory and locate the file afterward.
            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            self._output_path = None
            with open(self._log_path, "w") as log_file:
                process = subprocess.Popen(
                    [
                        "python", "/opt/seed-vc/inference.py",
                        "--source", input_path,
                        "--target", reference_path,
                        "--output", output_dir,
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                process.wait()
            if process.returncode != 0:
                self._status = "failed"
                self._output_path = None
            else:
                # inference.py names its output file
                # "vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"
                # (confirmed by reading inference.py's source directly), but
                # since we don't pass --length-adjust/--diffusion-steps/
                # --inference-cfg-rate ourselves, reconstructing that exact
                # string here would mean silently depending on upstream's
                # current default values. Instead, discover the file it
                # actually wrote: it is the only *.wav in the output
                # directory, or (if more than one somehow exists) the most
                # recently modified one.
                candidates = glob.glob(os.path.join(output_dir, "*.wav"))
                if candidates:
                    self._output_path = max(candidates, key=os.path.getmtime)
                    self._status = "done"
                else:
                    self._status = "failed"
                    self._output_path = None
        except Exception as exc:
            self._status = "failed"
            self._output_path = None
            try:
                with open(self._log_path, "a") as log_file:
                    log_file.write(f"\n[agent] _execute failed: {exc}\n")
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
