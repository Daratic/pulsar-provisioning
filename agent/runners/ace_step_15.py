import contextlib
import gc
import os
import re
import threading
import time
import traceback

from agent.runners.base import Runner, RunnerBusy
from agent.runners.lyric_score_lite import LATENT_FRAMES_PER_SECOND, SCORE_WINDOW_FRAMES, dit_lyric_score, latent_frames
# Imports librosa lazily, inside estimate_metas: a box without it gets no
# metas (logged by _estimate_metas) rather than an agent that cannot start.
from agent.runners.source_metas import estimate_metas

# Why the import failed, if it did. Without this the names below are simply
# None and the first call reports "'NoneType' object is not callable" - which
# is what jobs 21, 24 and 27 all reported, telling us nothing about the actual
# missing module. The reason is far more useful than the symptom.
IMPORT_ERROR: str | None = None

try:  # pragma: no cover - only importable where ACE-Step 1.5 is installed
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
except Exception as exc:  # not just ImportError: a dependency can raise
    # anything at import time, and swallowing it silently is how we lost three
    # jobs' worth of diagnosis. Tests patch these names directly, so their
    # values here don't matter locally.
    AceStepHandler = None
    LLMHandler = None
    GenerationParams = None
    GenerationConfig = None
    generate_music = None
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


PROJECT_ROOT = "/opt/ACE-Step-1.5"
CHECKPOINT_DIR = "/root/.cache/ace-step/checkpoints"
# XL turbo, not base. Upstream issue #1220 reports noise-only output from
# the API with SFT models and not with turbo, and a koboldcpp report says the
# same of 1.5 generally - which matches what we heard from base exactly:
# structurally valid audio that is not music. Turbo is the distilled model
# and the one people report as reliable.
#
# The 2B turbo is not published (its repo 401s); acestep-v15-xl-turbo is, and
# at 4B it sits inside a 5090's 32GB per the project's own VRAM tiers.
DEFAULT_DIT_MODEL = "acestep-v15-xl-turbo"

# Turbo is distilled for very few steps and bakes guidance in rather than
# using CFG - the model zoo lists it as "CFG: no, Step: 8", against base's
# "CFG: yes, Step: 50". Running turbo on base settings (64 steps, guidance
# 12) asks it for something it was never trained to do.
#
# shift is deliberately NOT set here. ACE-Step's own API documents it as
# "range 1.0~5.0, default 3.0. Only effective for base models, not turbo
# models" - so pinning it for turbo is inert. An earlier comment here had
# this exactly backwards, claiming 3.0 was turbo's schedule and 1.0 was
# base's; base is the one that wants 3.0, and jobs 72-74 ran base at 1.0.
TURBO_DEFAULTS = {"inference_steps": 8, "guidance_scale": 1.0}
DEFAULT_LM_MODEL = "acestep-5Hz-lm-0.6B"

# THE cover control, per ACE-Step's own guide: it sets what fraction of the
# denoising steps use the source audio's semantic codes rather than the
# text-only encoding. 1.0 means every step follows the source; lower means
# more steps are driven by the caption alone.
#
# The guide's bands: 0.3-0.5 dramatic genre change, 0.5-0.7 moderate,
# 0.7-0.9 subtle, 0.9-1.0 "same song, different take". 0.5 sits at the top
# of the dramatic band - a real restyle that still tracks the song.
DEFAULT_COVER_STRENGTH = 0.5

# ACE-Step's own default, and deliberately back to it.
#
# Cover does not work by starting the diffusion near the source. It quantises
# src_audio into semantic codes - melody, rhythm, chords, orchestration - and
# generates new audio conditioned on those codes plus the caption. The
# structure is carried by the codes; audio_cover_strength decides how much of
# the run follows them. ACE-Step's cover guidance gives no recommended value
# for this parameter at all, because it is not the one you reach for.
#
# It was set to 0.6 on the evidence of jobs 20, 33, 49 and 50 sounding like
# "chopped-up sounds, not music" at 0.0. Those jobs all predate the 5Hz LM
# and torch fixes, when every generation was noise regardless of settings -
# the diagnosis was made against a broken stack and then frozen into a
# default. It also silently truncates the schedule (0.6 left 3 of 8 steps),
# so every cover after it was a brief touch-up of the original and came back
# sounding exactly like the source: jobs 74, 78, 79 and 82.
#
# 0.0 is only this agent's fallback, for a caller that sends nothing. It is
# not what Pulsar sends: app/runner_schemas.py defaults cover_noise_strength
# to 0.15, and the UI and take sets fill schema defaults into every request,
# so a Pulsar cover runs at 0.15 unless the user changes it.
DEFAULT_COVER_NOISE_STRENGTH = 0.0

# Forwarded only when the job sets them, so anything unspecified keeps
# ACE-Step's own default instead of a value invented here. Names are those of
# the real GenerationParams dataclass, which differs from the REST layer's
# naming (caption/src_audio/duration, not prompt/src_audio_path/audio_duration).
OPTIONAL_PARAMS = (
    "inference_steps",
    "guidance_scale",
    "seed",
    "duration",
    "bpm",
    "keyscale",
    "timesignature",
    "vocal_language",
    "instrumental",
    "shift",
    "sampler_mode",
    "cover_noise_strength",
    "dcw_enabled",
    "latent_shift",
    "latent_rescale",
    "fade_in_duration",
    "fade_out_duration",
    "thinking",
    "use_cot_metas",
    "use_cot_caption",
    "use_cot_language",
    "instruction",
    "repainting_start",
    "repainting_end",
)

# Tasks that only work if the DiT actually sees the source audio. Both are
# Base-model-only (upstream docs/en/INFERENCE.md sections 4 and 6).
SOURCE_CONDITIONED_TASKS = ("complete", "lego")
LEGO_TRACKS = ("vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
               "percussion", "strings", "synth", "fx", "brass", "woodwinds")
DEFAULT_COMPLETE_TRACKS = "drums, bass, guitar, keyboard"


def _source_duration_seconds(path: str) -> float | None:
    """Length of the source in seconds, or None when it cannot be read.

    A take without a duration is still a take (the model then picks its own
    length), so this never raises.
    """
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception as exc:
        print(f"could not read the source duration of {path}: {exc}")
        return None

# Estimated from the source for a cover when the caller did not set them.
META_KEYS = ("bpm", "keyscale", "timesignature")

# ACE-Step logs the peak it is about to normalise away (acestep/inference.py,
# loguru, to stderr - which Supervisor writes into agent.log). A peak far
# below 1.0 is the August near-silence: 0.0041 amplified ~217x into "noise".
PEAK_LINE_RE = re.compile(r"\[Normalization\] Audio 0 BEFORE: Peak=([0-9]*\.?[0-9]+)")
# What get_lyric_score needs, from GenerationResult.extra_outputs. Each is
# batched; batch_size is 1, so sample 0 is this take.
ALIGNMENT_TENSOR_KEYS = (
    "pred_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents", "lyric_token_idss",
)
# The seed ACE-Step's own UI scores with, so our numbers compare with its.
ALIGNMENT_SEED = 42
# loguru -> stderr -> supervisor -> agent.log can land a moment after
# generate_music returns, so a missing peak line is read again before it is
# given up as None.
PEAK_READ_ATTEMPTS = 3
PEAK_READ_DELAY_SECONDS = 0.5


# What get_lyric_score allocated on top of the loaded models for a ~228 s
# source, from the OOM it raised in session 5 (2026-10-16). Logged beside the
# free memory before scoring, so a scoring OOM can be read at a glance.
SCORER_NEEDS_GIB = 8.25
GIB = 2 ** 30


def _torch():
    """torch if it is importable and has a GPU to talk to, else None."""
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _collect_and_empty_cache() -> None:
    """gc, then hand PyTorch's cached blocks back to the driver. Never raises."""
    gc.collect()
    try:
        torch = _torch()
        if torch is not None:
            torch.cuda.empty_cache()
    except Exception:
        pass


def _free_gpu_memory() -> None:
    """Give back what dropped handlers held on the GPU. Never raises.

    Dropping the last reference is not enough on its own: the model tensors
    can sit in reference cycles until gc runs, and PyTorch's caching allocator
    keeps freed blocks reserved until empty_cache.
    """
    _collect_and_empty_cache()


def _free_take_memory() -> None:
    """Give back what one take's generation and scoring left on the GPU.

    Runs after every take, success or failure. Session 5 on 2026-10-16 kept
    ~20-22 GiB allocated after a take - the models' ~12 GiB plus the take's
    leftovers - so the scorer could not get its 8.25 GiB and the next take
    could not generate at all. Separate from _free_gpu_memory, which is the
    release of the models themselves.
    """
    _collect_and_empty_cache()


def _gpu_memory_summary() -> str | None:
    """'allocated=X GiB free=Y GiB', or None without a GPU. Never raises."""
    try:
        torch = _torch()
        if torch is None:
            return None
        free, _total = torch.cuda.mem_get_info()
        return f"allocated={torch.cuda.memory_allocated() / GIB:.2f} GiB free={free / GIB:.2f} GiB"
    except Exception:
        return None


def _is_cuda_oom(error) -> bool:
    """A CUDA out-of-memory, given as an exception or as an error message.

    Both forms occur: generate_music and get_lyric_score catch RuntimeError -
    which torch.OutOfMemoryError is - and hand it back as text.
    """
    if error is None:
        return False
    if isinstance(error, BaseException) and any(
            cls.__name__ == "OutOfMemoryError" for cls in type(error).__mro__):
        return True
    return "out of memory" in str(error).lower()


def _is_progress_bar(line: str) -> bool:
    """tqdm output, which is most of agent.log by volume and carries nothing
    we ever read. The step count it implies is available from generate_music's
    own time_costs line, which survives this filter."""
    return "it/s" in line or "%|" in line or not line.strip()


class AceStep15Runner(Runner):
    """ACE-Step 1.5 in cover mode: regenerates an existing track in a new
    style while following its melodic contour, chord progression, rhythm and
    song form.

    Supersedes the v1 runner's audio2audio path, which has open upstream
    defects (ace-step/ACE-Step #287 "tends to create louder result, often
    distorts", #302 "audio2audio is broken in the latest codebase") and
    produced a full-scale-clipped, unmusical result on job 20 (2026-08-02).
    1.5 exposes `cover` as a first-class task type instead.

    ACE-Step skips its 5Hz LM automatically for cover/repaint/extract
    regardless of the `thinking` setting, so the LM handler is constructed
    but does no work here - generate_music still requires it positionally.
    """

    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")
        # One runner object serves a whole session, so these are reset at the
        # start of every take - a failed take must not show the last one's.
        self._scores: dict = {}
        # Source metas by (path, mtime): N takes of one source pay for the
        # librosa analysis once, and a replaced source is analysed again.
        self._metas_cache: dict[tuple[str, int], dict] = {}
        self._agent_log_path = os.path.join(workdir, "agent.log")
        self._sleep = time.sleep  # replaced in tests
        # The loaded DiT and LM, kept for the whole session and keyed by
        # (dit_model, lm_model, lm_backend). Building fresh handlers on every
        # take put one more ~12 GB copy on the GPU per take and never freed
        # the last one: session 3 on 2026-10-16 ran out of CUDA memory by its
        # third take. It also paid a full model load per take, which is the
        # cost a warm session exists to avoid.
        self._dit_handler = None
        self._llm_handler = None
        self._handler_key: tuple[str, str, str] | None = None
        # Whether THIS take is running on handlers an earlier take loaded. An
        # OOM on reused handlers means earlier takes left the GPU short, so
        # the handlers are released; an OOM on fresh ones is just a failure.
        self._reused_handlers = False
        # Guards the busy check and the thread start in submit together, so
        # two simultaneous requests cannot both see "not running".
        self._submit_lock = threading.Lock()

    def submit(self, input_path: str, params: dict) -> None:
        with self._submit_lock:
            # One take at a time. On 2026-10-16 a client that timed out and
            # re-posted /submit started a second _execute beside the first;
            # both loaded the model onto one GPU and one ended with meta
            # tensors. The agent answers RunnerBusy with 409.
            if self._status == "running":
                raise RunnerBusy("a take is already running on this runner")
            # Set synchronously, before the thread exists: a session reuses
            # this runner, and the previous job's "done" and output must not
            # be readable by the first poll after a new dispatch.
            self._status = "running"
            self._output_path = None
            self._scores = {}
            threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _release_handlers(self) -> None:
        """Drop the cached handlers and free the GPU memory they held."""
        dit, llm = self._dit_handler, self._llm_handler
        self._dit_handler = self._llm_handler = None
        self._handler_key = None
        del dit, llm
        _free_gpu_memory()

    def _handlers(self, params: dict):
        """The DiT and LM for this take: loaded once per model set, reused after.

        A failed initialisation is released, not cached, so the next take
        tries again from a clean GPU.
        """
        dit_model = params.get("dit_model", DEFAULT_DIT_MODEL)
        lm_model = params.get("lm_model", DEFAULT_LM_MODEL)
        lm_backend = params.get("lm_backend", "pt")
        key = (dit_model, lm_model, lm_backend)
        if self._handler_key == key and self._dit_handler is not None and self._llm_handler is not None:
            self._log(f"reusing loaded DiT/LM: dit_model={dit_model} lm_model={lm_model} backend={lm_backend}")
            self._reused_handlers = True
            return self._dit_handler, self._llm_handler
        if self._dit_handler is not None or self._llm_handler is not None:
            self._log(f"model set changed from {self._handler_key} to {key}: releasing the loaded DiT/LM")
            self._release_handlers()
        try:
            dit_handler, llm_handler = self._load_handlers(dit_model, lm_model, lm_backend)
        except Exception as exc:
            # Whatever the failed load left behind - possibly a half-loaded
            # model - must not stay on the GPU for the next take. The
            # traceback's frames still reference it, so clear them first.
            traceback.clear_frames(exc.__traceback__)
            self._release_handlers()
            raise
        self._dit_handler, self._llm_handler, self._handler_key = dit_handler, llm_handler, key
        return dit_handler, llm_handler

    def _execute(self, input_path: str, params: dict) -> None:
        """One take, then give back everything it put on the GPU.

        The take runs in _run_take so that its locals - the generation result,
        decoded audio, the scorer's inputs - die with that frame, before the
        cleanup in `finally` runs. A local of THIS frame would still be alive
        during the cleanup and keep its tensors.
        """
        self._status = "running"
        self._scores = {}
        self._reused_handlers = False
        try:
            self._run_take(input_path, params)
        except Exception as exc:
            self._status = "failed"
            self._output_path = None
            self._log(f"_execute failed: {exc}")
            oom = _is_cuda_oom(exc)
            # The traceback's frames hold the take's locals; clear them so
            # the cleanup below can free what they reference.
            traceback.clear_frames(exc.__traceback__)
            if oom:
                self._release_after_generation_oom()
        finally:
            _free_take_memory()
            summary = _gpu_memory_summary()
            if summary:
                self._log(f"GPU after take: {summary}")

    def _release_after_generation_oom(self) -> None:
        """A generation ran out of CUDA memory. On handlers an earlier take
        loaded, drop them so the next take reloads onto a clean GPU: one
        ~90 s reload instead of a chain of failed takes (takes 12 and 13 of
        session 5, 2026-10-16). On freshly loaded handlers, keep them."""
        if not self._reused_handlers:
            return
        self._log("generation ran out of CUDA memory on reused handlers: "
                  "releasing the loaded DiT/LM so the next take reloads them")
        self._release_handlers()

    def _run_take(self, input_path: str, params: dict) -> None:
        """Generate, then score. Raises on failure; _execute cleans up."""
        if AceStepHandler is None:
            raise RuntimeError(
                "ACE-Step 1.5 is installed but could not be imported by this runner. "
                f"Underlying import error: {IMPORT_ERROR}"
            )
        output_dir = os.path.join(self._workdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        dit_model = params.get("dit_model", DEFAULT_DIT_MODEL)
        task_type = params.get("task_type", "cover")
        # Refused before any model loads: a rejected job should cost nothing.
        lego_track = None
        if task_type in SOURCE_CONDITIONED_TASKS:
            if "turbo" in dit_model:
                raise ValueError(f"{task_type} needs a base DiT "
                                 f"(acestep-v15-base or acestep-v15-xl-base), not {dit_model}")
            if task_type == "lego":
                lego_track = params.get("track_name")
                if lego_track not in LEGO_TRACKS:
                    raise ValueError(f"lego needs track_name, one of {', '.join(LEGO_TRACKS)}")
        dit_handler, llm_handler = self._handlers(params)

        call_params = {
            # Overridable so text2music can be run as a control: if the
            # caption produces music from nothing but is ignored in cover
            # mode, the problem is the cover path, not the prompt.
            "task_type": task_type,
            # src_audio is the song being covered; reference_audio is the
            # STYLE it should take on. They are different inputs doing
            # different jobs, which the source docstrings say plainly:
            # reference_audio is "for style transfer or cover tasks",
            # src_audio is "for audio-to-audio tasks".
            #
            # Setting both to the same file - which this did briefly -
            # asks the model to make a track sound like itself, and it
            # duly returns something close to the original. Pointing
            # reference_audio at a track in the TARGET style is what
            # actually transfers a style.
            #
            # Falls back to the source when no style reference is given,
            # so a prompt-only cover still works.
            "caption": params.get("prompt") or "",
            # Left empty, ACE-Step renders the cover as an instrumental -
            # the exact mistake that made job 20's vocals unintelligible.
            "lyrics": params.get("lyrics") or "",
            # Stated rather than inherited: this is what stops a repeat of
            # job 20's clipping, so it should be visible at the call site.
            "enable_normalization": True,
            "normalization_db": -1.0,
        }
        # text2music generates from the caption alone. It is here as a
        # control - "can this stack make music from nothing?" - and
        # conditioning it on the very track under suspicion would defeat
        # that, so the audio inputs are omitted entirely rather than
        # passed and hopefully ignored.
        if task_type != "text2music":
            call_params["src_audio"] = input_path
            # Only when a real style reference was given. This used to
            # fall back to input_path, which set reference_audio ==
            # src_audio and asked the model to make the track sound like
            # itself - the trap this file already documents from job 74,
            # walked into by the fallback itself. Job 78 came back as the
            # source almost unchanged.
            #
            # With no reference, the caption is the only style signal,
            # which is what a prompt-driven cover is meant to be.
            if params.get("reference_path"):
                call_params["reference_audio"] = params["reference_path"]
            # Both of these only mean anything when there IS a source
            # audio. cover_noise_strength is documented as "0=pure noise
            # (no cover), 1=closest to src audio", and it also sets where
            # on the schedule the diffusion starts - at 0.6 only 40% of
            # the steps run.
            #
            # Sending it for text2music, which has no source, asked the
            # sampler to start 60% of the way towards silence and then
            # denoise for 3 of 8 steps. It returned very nearly what it
            # started with: job 76 decoded to Peak=0.0041, which
            # enable_normalization multiplied by ~217 to hit -1dB. That
            # is the "noise" - not a bad generation, but silence with the
            # gain turned all the way up.
            call_params["audio_cover_strength"] = params.get(
                "audio_cover_strength", DEFAULT_COVER_STRENGTH)
            if task_type not in SOURCE_CONDITIONED_TASKS:
                # A cover-schedule knob; meaningless for complete/lego.
                call_params["cover_noise_strength"] = params.get(
                    "cover_noise_strength", DEFAULT_COVER_NOISE_STRENGTH)

        if task_type in SOURCE_CONDITIONED_TASKS:
            # Job 67 (2026-10-16): complete ran with the 5Hz LM on. Upstream
            # logged "Audio codes provided, ignoring src_audio and using codes
            # instead" - the DiT never saw the vocal and the result was 258 s
            # of text-to-music. The LM is skipped only when ALL FOUR are off
            # (acestep/inference.py: use_lm = thinking or any use_cot_*).
            call_params["thinking"] = False
            call_params["use_cot_metas"] = False
            call_params["use_cot_caption"] = False
            call_params["use_cot_language"] = False
            if task_type == "complete":
                tracks = params.get("complete_tracks") or DEFAULT_COMPLETE_TRACKS
                call_params["instruction"] = f"Complete the input track with {tracks}:"
            else:
                call_params["instruction"] = f"Generate the {lego_track} track based on the audio context:"
            # Without a duration the model picks its own (258 s for a 229 s song).
            if "duration" not in params:
                seconds = _source_duration_seconds(input_path)
                if seconds is not None:
                    call_params["duration"] = seconds
                else:
                    self._log("source duration unreadable: leaving duration to the model")

        # Turbo's own schedule, unless the job explicitly says otherwise.
        if "turbo" in dit_model:
            call_params.update(TURBO_DEFAULTS)
        for optional in OPTIONAL_PARAMS:
            if optional in params:
                call_params[optional] = params[optional]
        if task_type in SOURCE_CONDITIONED_TASKS:
            # Job 70: the page default (0.15) rides in via params above; >0 re-noises
            # the SOURCE latents, so complete started from the vocal itself.
            if call_params.pop("cover_noise_strength", None) is not None:
                self._log("complete/lego: cover_noise_strength dropped (it re-noises from the source)")
            # Upstream PR #1266: DCW (default on) turns lego/complete output to noise.
            call_params.setdefault("dcw_enabled", False)
        # A cover always matches existing material, which is exactly when
        # ACE-Step's tutorial says to set these anchors. Only the ones the
        # caller left out; text2music has no source to measure.
        if task_type != "text2music":
            missing = [key for key in META_KEYS if key not in params]
            if missing:
                metas = self._estimate_metas(input_path)
                for key in missing:
                    if key in metas:
                        call_params[key] = metas[key]

        config = GenerationConfig(
            # Default is 2, i.e. two full renders per call. We pay by the
            # second and only ever retrieve one file.
            batch_size=1,
            # Default is flac; app/audio_quality.py reads 16-bit PCM wave,
            # and it gates delivery.
            audio_format="wav",
        )

        self._log("generation params: " + ", ".join(
            f"{k}={v!r}" for k, v in sorted(call_params.items()) if k != "lyrics"))
        self._log(f"lyrics: {len(call_params['lyrics'])} chars")
        log_offset = self._agent_log_size()
        result = generate_music(
            dit_handler, llm_handler, GenerationParams(**call_params), config, save_dir=output_dir
        )
        success = bool(getattr(result, "success", False))
        error = getattr(result, "error", "no error given")
        audios = getattr(result, "audios", None) or []
        self._log(f"generation returned success={getattr(result, 'success', None)!r} "
                  f"audios={len(audios)}")
        # Only the output path and the scorer's five tensors outlive the
        # result. Everything else it holds - decoded audio, the other
        # latents in extra_outputs - is dropped here, before the scorer
        # asks for its ~8 GiB (session 5 OOMed scoring with 20.6 GiB
        # still allocated).
        path = audios[0]["path"] if success and audios else None
        alignment_inputs = self._alignment_inputs(result, call_params) if path else None
        del result, audios

        if not success:
            self._log(f"generation reported failure: {error}")
            self._status = "failed"
            self._output_path = None
            if _is_cuda_oom(error):
                self._release_after_generation_oom()
            return

        # Trust the filesystem over the reported path: a job that claims
        # "done" with no retrievable audio is the worst outcome, because
        # the orchestrator then destroys the instance.
        if path and os.path.exists(path):
            # Scored BEFORE status flips: the worker reads the scores the
            # moment it sees "done". The peak is read last, after the file
            # is written and the alignment is scored, to give ACE-Step's
            # log line the most time to reach agent.log.
            alignment_score = self._alignment_score(dit_handler, alignment_inputs, call_params)
            del alignment_inputs
            self._scores = {
                "alignment_score": alignment_score,
                "peak_before_norm": self._read_peak(log_offset),
            }
            self._log(f"scores: {self._scores}")
            self._output_path = path
            self._status = "done"
        else:
            self._log(f"generation reported success but no file at {path!r}")
            self._status = "failed"
            self._output_path = None

    def _load_handlers(self, dit_model: str, lm_model: str, lm_backend: str):
        """Build and initialise a DiT and LM. Raises if either fails."""
        # Logged explicitly: a handler that fails to find its checkpoint
        # and initialises random weights produces correctly-shaped,
        # correctly-normalised noise - indistinguishable from a bad
        # generation unless we record what was actually loaded.
        self._log(f"init DiT project_root={PROJECT_ROOT} config_path={dit_model} device=cuda")
        # Two different directories, and confusing them cost us a lot of
        # time. initialize_service derives its own checkpoint dir as
        # project_root/checkpoints; CHECKPOINT_DIR is only ever where we
        # place the 5Hz LM by hand. Logging the LM's directory alone made
        # "1 entries" read as a missing DiT checkpoint when it was simply
        # a directory the DiT never looks at.
        dit_checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        self._log(f"DiT checkpoint dir {dit_checkpoint_dir}: {self._describe(dit_checkpoint_dir)}")
        self._log(f"LM checkpoint dir {CHECKPOINT_DIR}: {self._describe(CHECKPOINT_DIR)}")
        dit_handler = AceStepHandler()
        # initialize_service returns (message, ok) exactly like the LM's
        # initialize, and on a precheck failure it sets model, vae and
        # text_encoder to None and RETURNS rather than raising. Discarding
        # that is the same mistake that let the 5Hz LM silently not load:
        # generation then runs with nothing behind the handler and decodes
        # to near-silence, which enable_normalization amplifies into
        # full-scale noise. The message names the resolved model, VAE and
        # text-encoder paths, so it is also the only direct record of what
        # was actually loaded.
        dit_message, dit_ok = dit_handler.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=dit_model,
            device="cuda",
        )
        self._log(f"DiT initialise ok={dit_ok!r}: {dit_message}")
        if not dit_ok:
            raise RuntimeError(
                f"DiT failed to initialise (config_path={dit_model!r}): "
                f"{dit_message}. Generation without it produces noise."
            )
        llm_handler = LLMHandler()
        # initialize() returns (message, ok). Its result was previously
        # discarded, which let a failed LM initialisation pass silently -
        # and everything downstream then ran without the planning model.
        lm_message, lm_ok = llm_handler.initialize(
            checkpoint_dir=CHECKPOINT_DIR,
            lm_model_path=lm_model,
            # "pt", NOT "pytorch". The valid values are "vllm", "pt" and
            # "mlx"; anything else matches no branch, leaves
            # llm_initialized False, and returns ok=False. We passed
            # "pytorch" and discarded the result, so the 5Hz LM never
            # loaded at all - which is why text2music produced noise and
            # why the caption had no audible effect on covers
            # (generate_music logged need_lm_for_cot=True with
            # llm_initialized=False, so the chain-of-thought caption
            # processing was skipped entirely).
            backend=lm_backend,
            device="cuda",
        )
        self._log(f"LM initialise ok={lm_ok!r}: {lm_message}")
        if not lm_ok:
            # Loudly, rather than generating without the planning model.
            raise RuntimeError(
                f"5Hz LM failed to initialise (backend="
                f"{lm_backend!r}): {lm_message}. "
                "Generation without it produces noise."
            )
        return dit_handler, llm_handler

    def scores(self) -> dict:
        return dict(self._scores)

    def _agent_log_size(self) -> int:
        try:
            return os.path.getsize(self._agent_log_path)
        except OSError:
            return 0

    def _estimate_metas(self, input_path: str) -> dict:
        """Never raises: a take without anchors is still a take.

        Cached per (path, mtime). A failed estimate is not cached, so the next
        take tries again; a source whose mtime cannot be read is not cached.
        """
        try:
            key = (input_path, os.stat(input_path).st_mtime_ns)
        except OSError:
            key = None
        if key is not None and key in self._metas_cache:
            metas = self._metas_cache[key]
            self._log(f"metas reused for this source: {metas}")
            return dict(metas)
        try:
            metas = estimate_metas(input_path)
        except Exception as exc:
            self._log(f"metas: could not estimate from the source ({type(exc).__name__}: {exc})")
            return {}
        self._log(f"metas estimated from the source: {metas}")
        if key is not None:
            self._metas_cache[key] = dict(metas)
        return metas

    def _read_peak(self, offset: int) -> float | None:
        """_peak_since, retried briefly while agent.log exists but lacks the line."""
        for attempt in range(PEAK_READ_ATTEMPTS):
            peak = self._peak_since(offset)
            if peak is not None or not os.path.exists(self._agent_log_path):
                return peak
            if attempt < PEAK_READ_ATTEMPTS - 1:
                try:
                    self._sleep(PEAK_READ_DELAY_SECONDS)
                except Exception:
                    return None
        return None

    def _peak_since(self, offset: int) -> float | None:
        """The peak ACE-Step logged while making THIS take.

        agent.log carries every take of the session, so only what was written
        after `offset` is read, and the last match wins.
        """
        try:
            with open(self._agent_log_path, "rb") as log_file:
                size = os.fstat(log_file.fileno()).st_size
                log_file.seek(offset if offset <= size else 0)
                text = log_file.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        matches = PEAK_LINE_RE.findall(text)
        return float(matches[-1]) if matches else None

    def _alignment_inputs(self, result, call_params: dict) -> dict | None:
        """The five tensors get_lyric_score needs, taken out of the result so
        the result itself can be dropped before scoring. None when there is
        nothing to score. Never raises."""
        if not (call_params.get("lyrics") or "").strip():
            return None
        try:
            extra = getattr(result, "extra_outputs", None) or {}
            if any(extra.get(key) is None for key in ALIGNMENT_TENSOR_KEYS):
                self._log("alignment: extra_outputs lacks the tensors to score with")
                return None
            # Sample 0 of each batch: batch_size is 1, so this is the take.
            return {
                "pred_latent": extra["pred_latents"][0:1],
                "encoder_hidden_states": extra["encoder_hidden_states"][0:1],
                "encoder_attention_mask": extra["encoder_attention_mask"][0:1],
                "context_latents": extra["context_latents"][0:1],
                "lyric_token_ids": extra["lyric_token_idss"][0:1],
            }
        except Exception as exc:
            self._log(f"alignment: could not take the tensors to score with ({type(exc).__name__}: {exc})")
            return None

    def _alignment_score(self, dit_handler, inputs: dict | None, call_params: dict) -> float | None:
        """ACE-Step's lyric alignment score (its dit_score) for this take, or None.

        generate_music does not compute it. ACE-Step's own UI calls
        get_lyric_score, which scores a batch of two - pure noise and the take
        - and on a 24 GB card ran out of memory beside the loaded models (run
        6, 2026-10-16). dit_lyric_score is the same computation for the take
        alone. If even that runs out of memory, the first SCORE_WINDOW_FRAMES
        latent frames are scored once instead, and the log says so. Never
        raises - an unscored take is still a take, and an out-of-memory frees
        what it can and leaves the score None.
        """
        if inputs is None:
            return None
        torch = _torch()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
                free, _total = torch.cuda.mem_get_info()
                self._log(f"alignment: {free / GIB:.2f} GiB free on the GPU before scoring "
                          f"(ACE-Step's batch-of-two scorer needed {SCORER_NEEDS_GIB} GiB more than it got "
                          f"for a ~228 s source; this one scores the take alone)")
            except Exception:
                pass

        score, oom = self._score_once(dit_handler, inputs, call_params, torch, window_frames=None)
        if not oom:
            return score
        frames = latent_frames(inputs.get("pred_latent"))
        if frames is not None and frames <= SCORE_WINDOW_FRAMES:
            self._scoring_oom()
            return None
        _collect_and_empty_cache()
        seconds = SCORE_WINDOW_FRAMES / LATENT_FRAMES_PER_SECOND
        self._log(f"alignment: the whole take did not fit; scoring the first {SCORE_WINDOW_FRAMES} "
                  f"latent frames (~{seconds:.0f} s) instead")
        score, oom = self._score_once(dit_handler, inputs, call_params, torch, window_frames=SCORE_WINDOW_FRAMES)
        if oom:
            self._scoring_oom()
        elif score is not None:
            self._log(f"alignment: the score {score} is WINDOWED - it covers the first ~{seconds:.0f} s "
                      f"of the take, not all of it")
        return score

    def _score_once(self, dit_handler, inputs: dict, call_params: dict, torch,
                    window_frames: int | None) -> tuple[float | None, bool]:
        """(score or None, ran out of memory). Never raises."""
        span = "whole take" if window_frames is None else f"first {window_frames} frames"
        context = contextlib.nullcontext()
        if torch is not None:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            # dit_lyric_score enters inference_mode itself; stated here too
            # so nothing outside it keeps autograd state either.
            context = torch.inference_mode()
        try:
            with context:
                outcome = dit_lyric_score(
                    dit_handler,
                    **inputs,
                    vocal_language=call_params.get("vocal_language") or "en",
                    inference_steps=int(call_params.get("inference_steps", TURBO_DEFAULTS["inference_steps"])),
                    seed=ALIGNMENT_SEED,
                    window_frames=window_frames,
                )
        except Exception as exc:
            self._log(f"alignment: scoring the {span} raised {type(exc).__name__}: {exc}")
            oom = _is_cuda_oom(exc)
            traceback.clear_frames(exc.__traceback__)
            del exc
            self._log_scoring_peak(torch, span)
            return None, oom
        self._log_scoring_peak(torch, span)
        if not outcome.get("success"):
            error = outcome.get("error")
            self._log(f"alignment: scoring the {span} failed: {error}")
            return None, _is_cuda_oom(error)
        return float(outcome["dit_score"]), False

    def _log_scoring_peak(self, torch, span: str) -> None:
        """The most PyTorch held at once while scoring - the number that says
        how close the scorer came to the card's limit. Never raises."""
        if torch is None:
            return
        try:
            peak = torch.cuda.max_memory_allocated() / GIB
            self._log(f"alignment: peak allocated while scoring the {span}: {peak:.2f} GiB")
        except Exception:
            pass

    def _scoring_oom(self) -> None:
        self._log("alignment: out of GPU memory; the take is kept without a score")
        _collect_and_empty_cache()

    @staticmethod
    def _describe(path: str) -> str:
        """What is actually on disk where the checkpoint should be."""
        try:
            entries = sorted(os.listdir(path))
            return f"{len(entries)} entries: {entries[:8]}"
        except OSError as exc:
            return f"UNREADABLE ({exc})"

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

    def log_tail(self, lines: int = 600) -> str:
        """Our own log plus the agent process's stdout.

        ACE-Step prints its model loading to the process's stdout, which the
        provisioning script redirects to /workspace/agent.log - a different
        file from the one this runner writes. Reading only ours hid exactly
        the output needed to tell "loaded the checkpoint" from "quietly
        didn't".
        """
        # Bounded read: /workspace/agent.log carries the whole uvicorn +
        # model stdout and can be very large. Reading it whole made the
        # /logs endpoint drop the connection entirely (job 50), losing the
        # diagnostics this was added to collect.
        #
        # Read a wide window, then send a narrow one. At 64KB/100 lines the
        # window was narrower than a single generation, so the lines that
        # actually explain a run - notably
        #   [generate_audio] Cover mode: cover_noise_strength=0.6,
        #   effective_noise_level=0.4000, nearest_t=..., remaining_steps=3
        # - were pushed out by the tqdm bars that follow them. Simply
        # widening the window shipped the bars too, and the fetch then timed
        # out on a 4-minute job (job 78 captured no log at all).
        #
        # The bars are the bulk of the file and we never read them, so they
        # are dropped here, on the agent, before anything crosses the tunnel.
        # That keeps a whole generation's worth of real lines inside a
        # payload small enough to actually arrive.
        MAX_BYTES = 512 * 1024
        parts = []
        for label, path in (("runner", self._log_path), ("agent", "/workspace/agent.log")):
            try:
                with open(path, errors="replace") as log_file:
                    try:
                        log_file.seek(0, 2)
                        log_file.seek(max(0, log_file.tell() - MAX_BYTES))
                    except OSError:
                        pass
                    raw = log_file.read()
                # tqdm redraws with carriage returns, so one "line" can hold
                # hundreds of bars - split on them too before filtering.
                kept = [
                    segment for segment in raw.replace("\r", "\n").splitlines()
                    if not _is_progress_bar(segment)
                ]
                tail = "\n".join(kept[-lines:])
                if tail.strip():
                    parts.append("--- %s ---\n%s" % (label, tail))
            except OSError:
                continue
        return "\n".join(parts)
