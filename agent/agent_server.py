from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from agent.runners.base import Runner
from agent.runners.seed_vc import SeedVCRunner


class SubmitRequest(BaseModel):
    input_path: str
    params: dict


def create_agent_app(runner: Runner) -> FastAPI:
    agent_app = FastAPI(title="pulsar-agent")

    @agent_app.post("/submit")
    def submit(request: SubmitRequest) -> dict:
        runner.submit(request.input_path, request.params)
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
    def submit(request: MultiRunnerSubmitRequest) -> dict:
        _runner(request.runner).submit(request.input_path, request.params)
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

    return agent_app


# Entrypoint used by the agent Docker image's CMD (Task 10).
app = create_agent_app(SeedVCRunner())
