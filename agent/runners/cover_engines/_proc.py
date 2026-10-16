"""Running an engine's tool as a subprocess, its output in the job log."""

from __future__ import annotations

import subprocess

# How long a killed tool gets to exit before we stop waiting for it.
KILL_GRACE_SECONDS = 30


def run_tool(cmd: list[str], log, name: str, timeout_seconds: float, cwd: str | None = None,
             env: dict | None = None) -> None:
    """Run `cmd`, appending its stdout/stderr to the job log.

    Raises on a non-zero exit, naming the tool and the code, and on a hang:
    a tool still running after `timeout_seconds` (a stalled download, a
    wedged GPU) is killed rather than left to keep a rented box billing.
    """
    log(f"running {name} (timeout {timeout_seconds:.0f}s): {' '.join(cmd)}")
    with open(log.path, "a") as log_file:
        kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if env is not None:
            kwargs["env"] = env
        process = subprocess.Popen(cmd, **kwargs)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            log(f"{name} timed out after {timeout_seconds:.0f}s and was killed")
            raise RuntimeError(f"{name} timed out after {timeout_seconds:.0f}s (killed)")
    if process.returncode != 0:
        raise RuntimeError(f"{name} exited {process.returncode}")
