"""Entrypoint for the multi-runner agent.

Started by BYOMAgentAdapter on every rented instance. Replaces the old
one-module-per-model entrypoints (agent_server_demucs, agent_server_ace_step,
and agent_server's SeedVC default), each of which could serve exactly one
runner and so forced a separate GPU rental per model.

The registry builds runners lazily, so this imports cleanly on an instance
provisioned with only some of them - which is the normal case.
"""

from agent.agent_server import create_multi_runner_app
from agent.runners.registry import RunnerRegistry, default_factories

app = create_multi_runner_app(RunnerRegistry(default_factories()))
