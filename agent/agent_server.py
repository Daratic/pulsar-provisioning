import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from agent.runners.base import Runner, RunnerBusy
from agent.runners.seed_vc import SeedVCRunner


def _busy() -> JSONResponse:
    """A take is already running. 409, never a second take: a client that
    timed out and re-posted /submit on 2026-10-16 started two model loads on
    one GPU. The client reads 409 as "your earlier submit landed"."""
    return JSONResponse(status_code=409, content={"accepted": False, "reason": "busy"})


class SubmitRequest(BaseModel):
    input_path: str
    params: dict


def create_agent_app(runner: Runner) -> FastAPI:
    agent_app = FastAPI(title="pulsar-agent")

    @agent_app.post("/submit")
    def submit(request: SubmitRequest):
        try:
            runner.submit(request.input_path, request.params)
        except RunnerBusy:
            return _busy()
        return {"accepted": True}

    @agent_app.get("/status")
    def status() -> dict:
        return {"status": runner.status()}

    @agent_app.get("/logs", response_class=PlainTextResponse)
    def logs() -> str:
        return runner.log_tail()

    @agent_app.get("/output")
    def output() -> FileResponse:
        path = runner.output_path()
        if path is None:
            raise HTTPException(status_code=404, detail="Output not ready")
        return FileResponse(path)

    return agent_app


class MultiRunnerSubmitRequest(BaseModel):
    runner: str
    input_path: str
    params: dict


def create_multi_runner_app(registry) -> FastAPI:
    """Agent that hosts every runner installed on this instance.

    Supersedes create_agent_app's one-server-per-runner shape, which forced
    a whole GPU rental per model and made chaining two models impossible
    without paying to boot a second instance. Each endpoint names the runner
    it's addressing, so one job can drive several in sequence.
    """
    agent_app = FastAPI(title="pulsar-agent")

    def _runner(name: str):
        try:
            return registry.get(name)
        except KeyError as exc:
            # 404, not the 500 an uncaught KeyError would produce: asking for
            # a runner this instance wasn't provisioned with is a bad request,
            # not an agent fault, and the caller needs to tell them apart.
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @agent_app.get("/health")
    def health() -> dict:
        """Cheap liveness probe for the queue worker.

        A session's rental outlives any single job, so something has to notice
        a box that has died - a SIGBUS, a dropped link, a host gone
        unreachable - rather than leaving a queue waiting on it, still
        billing, until the hard cap fires.
        """
        return {"status": "ok", "runners": registry.names()}

    @agent_app.get("/runners")
    def runners() -> dict:
        return {"runners": registry.names()}

    @agent_app.post("/submit")
    def submit(request: MultiRunnerSubmitRequest):
        try:
            _runner(request.runner).submit(request.input_path, request.params)
        except RunnerBusy:
            return _busy()
        return {"accepted": True}

    @agent_app.get("/status")
    def status(runner: str) -> dict:
        return {"status": _runner(runner).status()}

    @agent_app.get("/logs", response_class=PlainTextResponse)
    def logs(runner: str) -> str:
        return _runner(runner).log_tail()

    @agent_app.get("/output-path")
    def output_path(runner: str) -> dict:
        # The path, not the bytes: the caller needs the file's real extension
        # to name what it retrieves (MIDI is not a .wav).
        return {"output_path": _runner(runner).output_path()}

    @agent_app.get("/scores")
    def scores(runner: str) -> dict:
        """Ranking signals for the runner's last output - {} if it has none."""
        return _runner(runner).scores()

    @agent_app.get("/output")
    def output(runner: str) -> FileResponse:
        path = _runner(runner).output_path()
        if path is None:
            raise HTTPException(status_code=404, detail="Output not ready")
        return FileResponse(path)

    @agent_app.get("/extra-outputs")
    def extra_outputs(runner: str) -> dict:
        """Names of the files a runner made besides its output (the cover
        pipeline's stems). Fetch each with /extra-output."""
        return {"names": sorted(_runner(runner).extra_outputs())}

    @agent_app.get("/extra-output")
    def extra_output(runner: str, name: str) -> FileResponse:
        # A bare file name only: the name comes off the network, and anything
        # with a separator or a dot-dot is refused before it is looked up.
        if not _is_basename(name):
            raise HTTPException(status_code=400, detail="name must be a plain file name")
        path = _runner(runner).extra_outputs().get(name)
        if path is None or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"No extra output {name!r}")
        return FileResponse(path)

    return agent_app


def _is_basename(name: str) -> bool:
    return (
        bool(name)
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
    )


# Entrypoint used by the agent Docker image's CMD (Task 10).
app = create_agent_app(SeedVCRunner())
