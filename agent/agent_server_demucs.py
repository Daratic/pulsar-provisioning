from agent.agent_server import create_agent_app
from agent.runners.demucs import DemucsRunner

app = create_agent_app(DemucsRunner())
