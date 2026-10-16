from abc import ABC, abstractmethod


class RunnerBusy(RuntimeError):
    """A submit arrived while this runner was already running a take.

    The agent answers it with 409. Starting a second take instead is what
    put two ~12 GB model loads on one GPU at once on 2026-10-16.
    """


class Runner(ABC):
    @abstractmethod
    def submit(self, input_path: str, params: dict) -> None:
        ...

    @abstractmethod
    def status(self) -> str:
        """Returns one of: 'idle', 'running', 'done', 'failed'."""

    @abstractmethod
    def output_path(self) -> str | None:
        ...

    @abstractmethod
    def log_tail(self, lines: int = 100) -> str:
        ...

    def scores(self) -> dict:
        """Signals for ranking this runner's last output. Most runners have none."""
        return {}

    def extra_outputs(self) -> dict[str, str]:
        """Files besides the main output, name -> path. Names are basenames
        (the agent serves them by name at /extra-output). Most runners have none."""
        return {}
