import os
import threading

from agent.runners.base import Runner

try:
    from acestep.pipeline_ace_step import ACEStepPipeline
except ImportError:  # pragma: no cover - only importable where the real
    # ACE-Step package is installed (the rented Vast.ai instance); tests
    # patch this name directly, so its value here doesn't matter locally.
    ACEStepPipeline = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - only importable where the real
    # soundfile package is installed (the rented Vast.ai instance); tests
    # patch this name directly, so its value here doesn't matter locally.
    sf = None


class AceStepRunner(Runner):
    """Drives ACE-Step's own pipeline class directly - unlike Seed-VC and
    Demucs, ACE-Step ships no standalone inference CLI script, only a
    Gradio UI and this Python library (github.com/ace-step/ACE-Step). Cover
    mode (audio2audio_enable=True) is the pipeline's own built-in
    audio-to-audio style-transfer path: it converts src_audio_path and
    ref_audio_input to latents internally, so no manual audio-encoding step
    is needed here.

    checkpoint_dir is hardcoded to /root/.cache/ace-step/checkpoints: the
    provisioning script's own `snapshot_download("ACE-Step/ACE-Step-v1-3.5B",
    cache_dir="/root/.cache/ace-step/checkpoints")` call primes that exact
    path at boot time, and constructing ACEStepPipeline with the same
    explicit path here means the runtime pipeline reads from the cache the
    provisioning script already populated - relying on each side's own
    default (as this used to) is what let them silently disagree and
    trigger a second, redundant ~7GB download at first inference.

    Does NOT pass src_audio_path - confirmed against ACE-Step's own source
    (2026-08-01) that it's unrelated to audio2audio: it's for continuing or
    editing a PREVIOUS ACE-Step generation (retake/repaint/extend/edit), and
    the pipeline's own internal validation (`assert task in ("repaint",
    "edit", "extend")` when src_audio_path is set) always fails for
    audio2audio, since audio2audio_enable + ref_audio_input auto-derive
    task="audio2audio" - not in that list. The actual audio2audio transform
    takes exactly one audio input: ref_audio_input.
    """

    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _execute(self, input_path: str, params: dict) -> None:
        # Runs on a daemon thread: an escaping exception would otherwise
        # leave status pinned at "idle" forever - see SeedVCRunner for the
        # same fix.
        self._status = "running"
        try:
            reference_path = params["reference_path"]
            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "cover.wav")

            audio_info = sf.info(input_path)
            source_duration = audio_info.frames / audio_info.samplerate

            call_kwargs = {
                "ref_audio_input": reference_path,
                "audio2audio_enable": True,
                # 0.5 half-ignores the source. Overridable per job because
                # the right value is entirely task-dependent: closer to 1.0
                # to stay near the original, lower to let the prompt lead.
                "ref_audio_strength": params.get("ref_audio_strength", 0.5),
                "prompt": params.get("prompt") or "",
                # Was hardcoded "" - ACE-Step treats an empty lyric sheet as
                # "instrumental", so any prompt asking for vocals produced
                # vocal-shaped noise instead of sung words (job 20).
                "lyrics": params.get("lyrics") or "",
                # Defaults to the source length, but overridable: ACE-Step
                # does not hold structure over several minutes in one shot,
                # and a full-length request is what job 20 asked for.
                "audio_duration": params.get("audio_duration", source_duration),
                "save_path": save_path,
            }
            # Forwarded ONLY when the job sets them, so anything we haven't
            # deliberately chosen falls through to ACE-Step's own defaults.
            # Sending a value guessed here would silently override a default
            # the library authors picked, which is worse than sending nothing.
            for optional in (
                "infer_step",
                "guidance_scale",
                "scheduler_type",
                "cfg_type",
                "omega_scale",
                # Fixing the seed makes a run reproducible - the only way to
                # tell "the prompt changed the output" from "the sampler
                # rolled differently" when tuning.
                "manual_seeds",
            ):
                if optional in params:
                    call_kwargs[optional] = params[optional]

            pipeline = ACEStepPipeline(checkpoint_dir="/root/.cache/ace-step/checkpoints")
            pipeline(**call_kwargs)

            if os.path.exists(save_path):
                self._output_path = save_path
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
