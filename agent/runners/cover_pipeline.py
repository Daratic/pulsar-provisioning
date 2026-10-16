"""cover_pipeline: a faithful cover of a song in one job.

Five stages on one daemon thread, one model on the GPU at a time:

  1 separate  source -> vocals.wav + instrumental.wav     (BS-RoFormer | demucs)
  2 analyse   bpm, key, time signature of the instrumental (librosa)
  3 backing   a new backing on the song's own melody/chords (MusicGen-Melody)
  4 voice     the vocal re-voiced from a reference clip    (Seed-VC | YingMusic-SVC | kept)
  5 mix       loudness-matched sum, -1 dBTP                (pyloudnorm)

Every parameter is checked before stage 1 (error prefix "params: "), so a
typo fails in a second instead of after a paid separation and MusicGen run.

The job output is cover.wav (backing.wav with `backing_only`). The stems are
extra outputs, served by the agent's /extra-output endpoint. A stage failure
fails the job with the stage's name in the error; whatever the earlier
stages made is kept and still listed.

Engines are chosen by params (`separator`, `backing_engine`, `voice_engine`,
`mixer`) from the registries in agent.runners.cover_engines. Nothing heavy is
imported here, so the registry can build this runner on any box.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
from dataclasses import dataclass

from agent.runners.base import Runner, RunnerBusy
from agent.runners.cover_engines import (
    BACKING_ENGINES,
    DEFAULT_BACKING_ENGINE,
    DEFAULT_MIXER,
    DEFAULT_SEPARATOR,
    DEFAULT_VOICE_ENGINE,
    MIXERS,
    SEPARATORS,
    VOICE_ENGINES,
    StageLog,
    collect_and_empty_cache,
)
from agent.runners.cover_engines.analysis import measure_bpm_key
from agent.runners.cover_engines.audio import audio_duration, fit_file_to_seconds
from agent.runners.cover_engines.voice import DIFFUSION_STEPS_RANGE, SEMITONE_RANGE, _int_param

STAGES = ("separate", "analyse", "backing", "voice", "mix")
MELODY_SOURCES = ("instrumental", "mix", "vocals")
KEEP_ORIGINAL = "keep_original"
# ACE-Step's range, and source_metas' clamp: outside it is a typo.
BPM_RANGE = (30.0, 300.0)
BEATS_PER_BAR_RANGE = (1, 16)
# A converted vocal this far off the stem's length is fitted back to it...
CONVERTED_LENGTH_TOLERANCE_SECONDS = 0.5
# ...and one shorter than this fraction of it was cut short: a failure.
CONVERTED_MIN_FRACTION = 0.5

_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


def free_memory() -> None:
    """Between stages: whatever the stage loaded goes back to the driver."""
    collect_and_empty_cache()


def reference_exists(path: str) -> bool:
    return os.path.isfile(path)


def _bool(value, name: str, default: bool) -> bool:
    """A JSON bool or its usual spellings; null or "" means the default."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _given(params: dict, key: str) -> bool:
    return params.get(key) not in (None, "")


def _number(params: dict, key: str, default: float) -> float:
    if not _given(params, key):
        return default
    try:
        return float(params[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {params[key]!r}")


def _timesig(value) -> int:
    # "4", 4 or "4/4" all mean four beats to the bar.
    try:
        beats = int(str(value).split("/")[0].strip())
    except ValueError:
        raise ValueError(f"timesignature must be beats per bar, like 4 or '4/4', got {value!r}")
    low, high = BEATS_PER_BAR_RANGE
    if not low <= beats <= high:
        raise ValueError(f"timesignature must have {low}-{high} beats per bar, got {value!r}")
    return beats


@dataclass(frozen=True)
class _Plan:
    """The job's parameters, checked."""

    caption: str
    reference: str | None
    keep_original: bool
    backing_only: bool
    vocal_level_db: float
    melody_source: str
    separator: str
    backing_engine: str
    voice_engine: str | None  # None with backing_only
    mixer: str
    bpm: float | None
    keyscale: str | None
    timesig: int | None


class _StageFailed(Exception):
    pass


class CoverPipelineRunner(Runner):
    def __init__(self, workdir: str = "/workspace", engines: dict | None = None):
        self._workdir = workdir
        self._job_dir = os.path.join(workdir, "cover")
        self._log_path = os.path.join(workdir, "job.log")
        self._engines = engines or {
            "separator": SEPARATORS,
            "backing_engine": BACKING_ENGINES,
            "voice_engine": VOICE_ENGINES,
            "mixer": MIXERS,
        }
        self._status = "idle"
        self._output_path: str | None = None
        # Written by the job thread, read by the agent's request threads.
        self._extra: dict[str, str] = {}
        self._extra_lock = threading.Lock()
        self.notes: list[str] = []
        self.error: str | None = None
        self._submit_lock = threading.Lock()

    # --- Runner interface ---------------------------------------------------

    def submit(self, input_path: str, params: dict) -> None:
        with self._submit_lock:
            # One job at a time: two would put two models on one GPU.
            if self._status == "running":
                raise RunnerBusy("a cover is already running on this runner")
            # Set before the thread exists: the previous job's "done", output
            # and stems must not be readable by the first poll of this one.
            self._status = "running"
            self._output_path = None
            self._reset_extra()
            threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def status(self) -> str:
        return self._status

    def output_path(self) -> str | None:
        return self._output_path

    def extra_outputs(self) -> dict[str, str]:
        """The stems made so far, by name: a copy. Only files inside the
        workdir, because the agent serves whatever this returns."""
        with self._extra_lock:
            items = list(self._extra.items())
        root = os.path.realpath(self._workdir)
        inside = {}
        for name, path in items:
            real = os.path.realpath(path)
            try:
                if os.path.commonpath([root, real]) == root:
                    inside[name] = path
            except ValueError:  # different drives on Windows
                continue
        return inside

    def log_tail(self, lines: int = 100) -> str:
        try:
            with open(self._log_path) as log_file:
                return "".join(log_file.readlines()[-lines:])
        except FileNotFoundError:
            return ""

    # --- the job ------------------------------------------------------------

    def _execute(self, input_path: str, params: dict) -> None:
        # Runs on a daemon thread: an escaping exception would leave status
        # pinned at "running" for ever, so everything is caught - even
        # BaseException, which a library's sys.exit() raises.
        self._status = "running"
        self._output_path = None
        self._reset_extra()
        self.notes = []
        self.error = None
        try:
            open(self._log_path, "w").close()
            shutil.rmtree(self._job_dir, ignore_errors=True)
            os.makedirs(self._job_dir, exist_ok=True)
            plan = self._plan(params)
            self._output_path = self._run_stages(input_path, plan, params)
            self._status = "done"
        except _StageFailed as exc:
            self.error = str(exc)
            self._output_path = None
            self._status = "failed"
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._output_path = None
            self._status = "failed"
            self._log(f"cover failed: {self.error}\n{traceback.format_exc()}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            free_memory()

    def _plan(self, params: dict) -> _Plan:
        try:
            return self._check_params(params)
        except ValueError as exc:
            self._log(f"params rejected: {exc}")
            raise _StageFailed(f"params: {exc}") from None

    def _check_params(self, params: dict) -> _Plan:
        caption = str(params.get("style") or params.get("caption") or "").strip()
        if not caption:
            raise ValueError("style (the caption for the new backing) is required")
        reference = params.get("reference_path") or None
        keep_original = _bool(params.get("keep_original_vocals"), "keep_original_vocals", reference is None)
        backing_only = _bool(params.get("backing_only"), "backing_only", False)
        vocal_level_db = _number(params, "vocal_level_db", 0.0)

        names = {}
        for kind, default in (("separator", DEFAULT_SEPARATOR), ("backing_engine", DEFAULT_BACKING_ENGINE),
                              ("voice_engine", DEFAULT_VOICE_ENGINE), ("mixer", DEFAULT_MIXER)):
            name = params.get(kind) or default
            if name not in self._engines[kind]:
                raise ValueError(f"unknown {kind} {name!r}; known: {sorted(self._engines[kind])}")
            names[kind] = name

        melody_source = params.get("melody_source") or "instrumental"
        if melody_source not in MELODY_SOURCES:
            raise ValueError(f"melody_source must be one of {list(MELODY_SOURCES)}, got {melody_source!r}")

        bpm = None
        if _given(params, "bpm"):
            bpm = _number(params, "bpm", 0.0)
            low, high = BPM_RANGE
            if not low <= bpm <= high:
                raise ValueError(f"bpm must be between {low:.0f} and {high:.0f}, got {params['bpm']!r}")
        timesig = _timesig(params["timesignature"]) if _given(params, "timesignature") else None
        keyscale = str(params["keyscale"]) if _given(params, "keyscale") else None

        _int_param(params, "semitone_shift", 0, SEMITONE_RANGE)
        _int_param(params, "diffusion_steps", 30, DIFFUSION_STEPS_RANGE)
        if _number(params, "window_seconds", 30.0) <= 0:
            raise ValueError(f"window_seconds must be positive, got {params['window_seconds']!r}")
        _number(params, "guidance_scale", 3.0)
        if _given(params, "seed"):
            try:
                int(str(params["seed"]).strip())
            except ValueError:
                raise ValueError(f"seed must be a whole number, got {params['seed']!r}")

        if backing_only:
            voice = None
        elif reference is None:
            if not keep_original:
                raise ValueError("no reference clip (reference_path) to convert to, and keep_original_vocals is false")
            voice = KEEP_ORIGINAL
        else:
            voice = names["voice_engine"]
            if voice != KEEP_ORIGINAL and not reference_exists(reference):
                raise ValueError(f"reference_path {reference!r} is not a file on this box")

        return _Plan(
            caption=caption, reference=reference, keep_original=keep_original, backing_only=backing_only,
            vocal_level_db=vocal_level_db, melody_source=melody_source, separator=names["separator"],
            backing_engine=names["backing_engine"], voice_engine=voice, mixer=names["mixer"],
            bpm=bpm, keyscale=keyscale, timesig=timesig,
        )

    def _run_stages(self, input_path: str, plan: _Plan, params: dict) -> str:
        log = StageLog(self._log_path)
        job = self._job_dir

        # 1 separate
        vocals, instrumental = self._stage(
            1, lambda: self._engine("separator", plan.separator).run(input_path, job, log)
        )
        self._add_extra({"vocals.wav": vocals, "instrumental.wav": instrumental})

        # 2 analyse
        bpm, keyscale, timesig, target_seconds = self._stage(2, lambda: self._analyse(instrumental, plan, log))

        # 3 backing
        melody = {"instrumental": instrumental, "mix": input_path, "vocals": vocals}[plan.melody_source]
        backing = self._stage(3, lambda: self._engine("backing_engine", plan.backing_engine).run(
            melody, plan.caption, bpm, keyscale, timesig, target_seconds, params, job, log
        ))
        self._add_extra({"backing.wav": backing})

        if plan.backing_only:
            self._log_stage(4, "skipped (backing_only)")
            self._log_stage(5, "skipped (backing_only)")
            return backing

        # 4 voice
        voiced, converted = self._stage(4, lambda: self._voice(vocals, plan, target_seconds, params, log))
        if converted:
            self._add_extra({"vocals_converted.wav": voiced})

        # 5 mix
        def mix_stage():
            out = self._engine("mixer", plan.mixer).run(
                voiced, backing, plan.vocal_level_db, os.path.join(job, "cover.wav"), log
            )
            if not os.path.exists(out):
                raise RuntimeError(f"the mixer returned {out} but wrote nothing there")
            return out

        return self._stage(5, mix_stage)

    def _analyse(self, instrumental: str, plan: _Plan, log: StageLog):
        # The length comes from the separator's WAV, not the upload: the
        # separator reads formats (m4a, some mp3) that soundfile cannot, and
        # the stem is the same length as the source.
        target_seconds = audio_duration(instrumental)
        if plan.bpm is not None and plan.keyscale is not None and plan.timesig is not None:
            bpm, keyscale, timesig = plan.bpm, plan.keyscale, plan.timesig
            log("analysis: bpm, key and time signature all given; nothing measured")
        else:
            bpm, keyscale, timesig = measure_bpm_key(instrumental)
            log(f"analysis: measured {bpm:.0f} bpm, {keyscale}, {timesig} beats/bar from the instrumental")
            bpm = plan.bpm if plan.bpm is not None else bpm
            keyscale = plan.keyscale if plan.keyscale is not None else keyscale
            timesig = plan.timesig if plan.timesig is not None else timesig
        log(f"analysis: using {bpm:.0f} bpm, {keyscale}, {timesig} beats/bar; source {target_seconds:.2f}s")
        return bpm, keyscale, timesig, target_seconds

    def _voice(self, vocals: str, plan: _Plan, target_seconds: float, params: dict, log: StageLog):
        if plan.voice_engine == KEEP_ORIGINAL:
            if plan.reference is None:
                self._note("no reference clip was given: the original vocals are kept")
            else:
                self._note("voice_engine keep_original: the original vocals are kept")
            return self._engine("voice_engine", KEEP_ORIGINAL).run(vocals, None, params, self._job_dir, log)
        try:
            path, converted = self._engine("voice_engine", plan.voice_engine).run(
                vocals, plan.reference, params, self._job_dir, log
            )
            if converted:
                self._check_converted_length(path, target_seconds, log)
            return path, converted
        except Exception as exc:
            if not plan.keep_original:
                raise
            self._note(f"voice conversion ({plan.voice_engine}) failed: {exc}; the original vocals are kept")
            return vocals, False

    def _check_converted_length(self, path: str, stem_seconds: float, log: StageLog) -> None:
        """A converter that exits 0 with a short file must not reach the mix
        silently: the vocal would just stop."""
        got = audio_duration(path)
        if got < CONVERTED_MIN_FRACTION * stem_seconds:
            raise RuntimeError(
                f"the converted vocal is {got:.2f}s, under half the {stem_seconds:.2f}s stem: it was cut short"
            )
        if abs(got - stem_seconds) > CONVERTED_LENGTH_TOLERANCE_SECONDS:
            log(f"warning: the converted vocal is {got:.2f}s against the stem's {stem_seconds:.2f}s; "
                "trimmed/padded to the stem")
            fit_file_to_seconds(path, stem_seconds)

    # --- plumbing -----------------------------------------------------------

    def _engine(self, kind: str, name: str):
        return self._engines[kind][name]()

    def _reset_extra(self) -> None:
        with self._extra_lock:
            self._extra = {}

    def _add_extra(self, entries: dict[str, str]) -> None:
        with self._extra_lock:
            self._extra = {**self._extra, **entries}

    def _stage(self, number: int, body):
        """Run one stage: time it, log it, free memory after it, and turn
        any failure into one that names the stage."""
        name = STAGES[number - 1]
        began = time.monotonic()
        try:
            result = body()
        except Exception as exc:
            elapsed = time.monotonic() - began
            message = f"stage {number}/5 {name} failed: {type(exc).__name__}: {exc}"
            self._log(f"stage {number}/5 {name} failed after {elapsed:.1f}s: {type(exc).__name__}: {exc}\n"
                      f"{traceback.format_exc()}")
            raise _StageFailed(message) from None
        finally:
            free_memory()
        self._log_stage(number, f"done in {time.monotonic() - began:.1f}s")
        return result

    def _log_stage(self, number: int, what: str) -> None:
        self._log(f"stage {number}/5 {STAGES[number - 1]} {what}")

    def _note(self, message: str) -> None:
        self.notes.append(message)
        self._log(f"note: {message}")

    def _log(self, message: str) -> None:
        StageLog(self._log_path)(message)
