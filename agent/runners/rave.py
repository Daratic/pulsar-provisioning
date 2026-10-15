import os
import threading

from agent.runners.base import Runner

try:  # pragma: no cover - only importable on a provisioned instance
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
except ImportError:  # tests patch these names directly
    sf = None
    torch = None
    hf_hub_download = None


MODEL_DIR = "/opt/rave/models"
# RAVE models are trained at a fixed rate, commonly 48kHz. Writing the output
# at the source's rate instead would pitch-shift and time-stretch it.
DEFAULT_SAMPLE_RATE = 48000


class RaveRunner(Runner):
    """Timbre transfer with RAVE (ACIDS-IRCAM).

    Turns one sound into another while keeping the performance: a guitar line
    played back as saxophone, a single vocal rendered as a choir. The
    mechanism is the model's own autoencoder - encode the source audio, then
    decode it through a model trained on the target instrument. The musical
    gestures survive; the timbre is replaced.

    RAVE rather than Magenta's DDSP, which is the better-known option: DDSP
    is monophonic-only, and both DDSP VST and MIDI-DDSP were archived in
    2024. RAVE is maintained, generates at 48kHz, handles polyphonic and
    noisy material, and runs 20-80x faster than realtime.

    Models are TorchScript (.ts) files, so no RAVE source install is needed
    at inference time - torch.jit.load is enough.
    """

    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _resolve_model(self, params: dict) -> str | None:
        if params.get("model_path"):
            return params["model_path"]
        if params.get("model_repo") and params.get("model_file"):
            return hf_hub_download(params["model_repo"], params["model_file"], local_dir=MODEL_DIR)
        return None

    def _execute(self, input_path: str, params: dict) -> None:
        self._status = "running"
        try:
            model_path = self._resolve_model(params)
            if not model_path:
                self._log(
                    "no target timbre given - set params.model_path, or "
                    "model_repo + model_file to pull a RAVE .ts model from HuggingFace"
                )
                self._status = "failed"
                self._output_path = None
                return

            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "timbre.wav")

            model = torch.jit.load(model_path)
            model.eval()

            audio, _source_rate = sf.read(input_path, dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                # RAVE models are mono; averaging keeps both channels'
                # content rather than silently discarding one.
                audio = audio.mean(axis=1)

            with torch.no_grad():
                x = torch.from_numpy(audio).reshape(1, 1, -1)
                latent = model.encode(x)
                # Latent manipulation goes here for more adventurous effects;
                # a straight round-trip is the timbre transfer itself.
                out = model.decode(latent)

            rendered = out.squeeze().cpu().numpy()
            sf.write(output_path, rendered, params.get("sample_rate", DEFAULT_SAMPLE_RATE))

            if os.path.exists(output_path):
                self._output_path = output_path
                self._status = "done"
            else:
                self._log(f"RAVE produced no file at {output_path}")
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
