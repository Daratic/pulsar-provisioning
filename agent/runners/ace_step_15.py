import os
import re
import threading
import time

from agent.runners.base import Runner
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
    "latent_shift",
    "latent_rescale",
    "fade_in_duration",
    "fade_out_duration",
)

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

    def submit(self, input_path: str, params: dict) -> None:
        # Set synchronously, before the thread exists: a session reuses this
        # runner, and the previous job's "done" and output must not be
        # readable by the first poll after a new dispatch.
        self._status = "running"
        self._output_path = None
        self._scores = {}
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _execute(self, input_path: str, params: dict) -> None:
        self._status = "running"
        self._scores = {}
        try:
            if AceStepHandler is None:
                raise RuntimeError(
                    "ACE-Step 1.5 is installed but could not be imported by this runner. "
                    f"Underlying import error: {IMPORT_ERROR}"
                )
            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            dit_model = params.get("dit_model", DEFAULT_DIT_MODEL)
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
                lm_model_path=params.get("lm_model", DEFAULT_LM_MODEL),
                # "pt", NOT "pytorch". The valid values are "vllm", "pt" and
                # "mlx"; anything else matches no branch, leaves
                # llm_initialized False, and returns ok=False. We passed
                # "pytorch" and discarded the result, so the 5Hz LM never
                # loaded at all - which is why text2music produced noise and
                # why the caption had no audible effect on covers
                # (generate_music logged need_lm_for_cot=True with
                # llm_initialized=False, so the chain-of-thought caption
                # processing was skipped entirely).
                backend=params.get("lm_backend", "pt"),
                device="cuda",
            )
            self._log(f"LM initialise ok={lm_ok!r}: {lm_message}")
            if not lm_ok:
                # Loudly, rather than generating without the planning model.
                raise RuntimeError(
                    f"5Hz LM failed to initialise (backend="
                    f"{params.get('lm_backend', 'pt')!r}): {lm_message}. "
                    "Generation without it produces noise."
                )

            task_type = params.get("task_type", "cover")
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
                call_params["cover_noise_strength"] = params.get(
                    "cover_noise_strength", DEFAULT_COVER_NOISE_STRENGTH)

            # Turbo's own schedule, unless the job explicitly says otherwise.
            if "turbo" in dit_model:
                call_params.update(TURBO_DEFAULTS)
            for optional in OPTIONAL_PARAMS:
                if optional in params:
                    call_params[optional] = params[optional]
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
            self._log(f"generation returned success={getattr(result, 'success', None)!r} "
                      f"audios={len(getattr(result, 'audios', None) or [])}")

            if not getattr(result, "success", False):
                self._log(f"generation reported failure: {getattr(result, 'error', 'no error given')}")
                self._status = "failed"
                self._output_path = None
                return

            audios = getattr(result, "audios", None) or []
            path = audios[0]["path"] if audios else None
            # Trust the filesystem over the reported path: a job that claims
            # "done" with no retrievable audio is the worst outcome, because
            # the orchestrator then destroys the instance.
            if path and os.path.exists(path):
                # Scored BEFORE status flips: the worker reads the scores the
                # moment it sees "done". The peak is read last, after the file
                # is written and the alignment is scored, to give ACE-Step's
                # log line the most time to reach agent.log.
                alignment_score = self._alignment_score(dit_handler, result, call_params)
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
        except Exception as exc:
            self._status = "failed"
            self._output_path = None
            self._log(f"_execute failed: {exc}")

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

    def _alignment_score(self, dit_handler, result, call_params: dict) -> float | None:
        """ACE-Step's lyric alignment score for this take, or None.

        generate_music does not compute it. ACE-Step's own UI does, by passing
        the result's extra_outputs to get_lyric_score; its dit_score scores the
        generated latent (lm_score is the pure-noise baseline). Never raises -
        an unscored take is still a take.
        """
        if not (call_params.get("lyrics") or "").strip():
            return None
        try:
            extra = getattr(result, "extra_outputs", None) or {}
            if any(extra.get(key) is None for key in ALIGNMENT_TENSOR_KEYS):
                self._log("alignment: extra_outputs lacks the tensors to score with")
                return None
            outcome = dit_handler.get_lyric_score(
                pred_latent=extra["pred_latents"][0:1],
                encoder_hidden_states=extra["encoder_hidden_states"][0:1],
                encoder_attention_mask=extra["encoder_attention_mask"][0:1],
                context_latents=extra["context_latents"][0:1],
                lyric_token_ids=extra["lyric_token_idss"][0:1],
                vocal_language=call_params.get("vocal_language") or "en",
                inference_steps=int(call_params.get("inference_steps", TURBO_DEFAULTS["inference_steps"])),
                seed=ALIGNMENT_SEED,
            )
            if not outcome.get("success"):
                self._log(f"alignment: get_lyric_score failed: {outcome.get('error')}")
                return None
            return float(outcome["dit_score"])
        except Exception as exc:
            self._log(f"alignment: scoring raised {type(exc).__name__}: {exc}")
            return None

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
