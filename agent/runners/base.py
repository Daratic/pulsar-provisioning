from abc import ABC, abstractmethod


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
