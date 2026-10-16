"""Stage engines for the cover_pipeline runner.

Each stage of a cover (separate, backing, voice, mix) has an interface and
one or more engines. An engine is one small class with a `run(...)` method
and an `INSTALL` constant naming what a box needs to run it. The catalogue
(app/model_adapters/runner_catalog.py) installs every engine's INSTALL, so
adding an engine is one class here plus its registry entry below.

Nothing here imports numpy, torch or any audio library at module level: the
agent imports this on boxes without them, and Pulsar's tests run without them.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineInstall:
    """What a box needs for one engine. Consumed by the catalogue.

    pip_packages: installed with the shared `pip install` of the runner set.
    setup_commands: shell commands run after pip (clones, venvs, requirements).
    predownload_commands: shell commands that fetch weights at install time,
    so a job never waits on (or fails at) a download.
    """

    pip_packages: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    predownload_commands: tuple[str, ...] = ()


# For the stages that are not engines: analysis (librosa bpm/key) and the
# audio helpers (soundfile, scipy resampling).
PIPELINE_INSTALL = EngineInstall(pip_packages=("librosa==0.11.0", "soundfile", "scipy"))

# ffmpeg decodes compressed sources; sox is what YingMusic-SVC's
# torchaudio.sox_effects runs. Two lines, not an && chain: under `set -e` a
# failed `apt-get update` in a non-final position would not stop the script.
AUDIO_TOOLS_APT = (
    "apt-get update",
    "apt-get install -y --no-install-recommends sox libsox-fmt-all ffmpeg",
)


class StageLog:
    """The job log, as engines see it.

    Calling it appends a `[cover]` line. Subprocess engines open `path` in
    append mode and hand it to the tool as stdout, so the tool's own output
    lands in the same job.log the agent serves at /logs.
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, message: str) -> None:
        try:
            with open(self.path, "a") as log_file:
                log_file.write(f"[cover] {message}\n")
        except OSError:
            pass


class Separator:
    INSTALL = EngineInstall()

    def run(self, src: str, workdir: str, log: StageLog) -> tuple[str, str]:
        """Returns (vocals_path, instrumental_path)."""
        raise NotImplementedError


class BackingEngine:
    INSTALL = EngineInstall()

    def run(self, melody_stem: str, caption: str, bpm: float, keyscale: str, timesig: int,
            target_seconds: float, params: dict, workdir: str, log: StageLog) -> str:
        raise NotImplementedError


class VoiceEngine:
    INSTALL = EngineInstall()

    def run(self, vocals: str, reference: str | None, params: dict, workdir: str,
            log: StageLog) -> tuple[str, bool]:
        """Returns (path, converted)."""
        raise NotImplementedError


class Mixer:
    INSTALL = EngineInstall()

    def run(self, vocals: str, backing: str, vocal_level_db: float, out_path: str, log: StageLog) -> str:
        raise NotImplementedError


def _torch_with_gpu():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def collect_and_empty_cache() -> None:
    """gc, then hand PyTorch's cached blocks back to the driver. Never raises.

    Same idiom as ace_step_15: dropping the last reference is not enough,
    because model tensors can sit in reference cycles until gc runs and the
    caching allocator keeps freed blocks reserved until empty_cache.
    """
    gc.collect()
    try:
        torch = _torch_with_gpu()
        if torch is not None:
            torch.cuda.empty_cache()
    except Exception:
        pass


# Registries come last: the engine modules import the base classes above.
from agent.runners.cover_engines.backing import MusicgenMelodyBacking  # noqa: E402
from agent.runners.cover_engines.mixer import LoudnessMixer  # noqa: E402
from agent.runners.cover_engines.separators import BsRoformerSeparator, DemucsSeparator  # noqa: E402
from agent.runners.cover_engines.voice import KeepOriginalVoice, SeedVcVoice, YingMusicSvcVoice  # noqa: E402

SEPARATORS = {"bs_roformer": BsRoformerSeparator, "demucs": DemucsSeparator}
BACKING_ENGINES = {"musicgen_melody": MusicgenMelodyBacking}
VOICE_ENGINES = {"seed_vc": SeedVcVoice, "yingmusic_svc": YingMusicSvcVoice, "keep_original": KeepOriginalVoice}
MIXERS = {"loudness": LoudnessMixer}

DEFAULT_SEPARATOR = "bs_roformer"
DEFAULT_BACKING_ENGINE = "musicgen_melody"
DEFAULT_VOICE_ENGINE = "seed_vc"
DEFAULT_MIXER = "loudness"


def all_engine_classes() -> list[type]:
    """Every engine, for the catalogue to collect INSTALL constants from."""
    return [*SEPARATORS.values(), *BACKING_ENGINES.values(), *VOICE_ENGINES.values(), *MIXERS.values()]
