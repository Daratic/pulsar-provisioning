"""Which runners this particular instance can actually run.

Each runner imports its own model library (acestep, demucs, rvc, basic_pitch),
and an instance is only provisioned with the subset the job asked for. So
runners are built lazily: importing every runner eagerly would crash the
agent on boot for any instance that wasn't given all of them, which is now
the normal case rather than the exception.
"""

from typing import Callable

from agent.runners.base import Runner


class RunnerRegistry:
    def __init__(self, factories: dict[str, Callable[[], Runner]]):
        self._factories = factories
        self._instances: dict[str, Runner] = {}

    def names(self) -> list[str]:
        return list(self._factories)

    def get(self, name: str) -> Runner:
        if name not in self._factories:
            raise KeyError(
                f"Runner {name!r} is not installed on this instance. Available: {sorted(self._factories)}"
            )
        # Cached: a runner holds the job's status and output path, so a second
        # construction would silently reset an in-flight job back to "idle".
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
        return self._instances[name]


def default_factories(workdir: str = "/workspace") -> dict[str, Callable[[], Runner]]:
    """Every runner Pulsar knows about, as import-on-demand factories.

    The imports are deliberately inside the lambdas: at module import time
    none of the model libraries are touched, so this is safe to call on an
    instance provisioned with only one of them.
    """

    def _demucs():
        from agent.runners.demucs import DemucsRunner

        return DemucsRunner(workdir=workdir)

    def _seed_vc():
        from agent.runners.seed_vc import SeedVCRunner

        return SeedVCRunner(workdir=workdir)

    def _ace_step():
        from agent.runners.ace_step import AceStepRunner

        return AceStepRunner(workdir=workdir)

    def _ace_step_15():
        from agent.runners.ace_step_15 import AceStep15Runner

        return AceStep15Runner(workdir=workdir)

    def _rvc():
        from agent.runners.rvc import RVCRunner

        return RVCRunner(workdir=workdir)

    def _rave():
        from agent.runners.rave import RaveRunner

        return RaveRunner(workdir=workdir)

    def _audio_to_midi():
        from agent.runners.audio_to_midi import AudioToMidiRunner

        return AudioToMidiRunner(workdir=workdir)

    return {
        "demucs": _demucs,
        "seed_vc": _seed_vc,
        "ace_step": _ace_step,
        "ace_step_15": _ace_step_15,
        "rvc": _rvc,
        "rave": _rave,
        "audio_to_midi": _audio_to_midi,
    }
