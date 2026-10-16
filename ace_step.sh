#!/bin/bash
# Generated from Pulsar's app/model_adapters/runner_catalog.py by
# scripts/generate_provisioning.py. Do not edit by hand: change the
# catalog, regenerate, export, and move Pulsar's pin.
# Runner set: ace_step
set -e
mkdir -p /workspace
if [ -f /venv/main/bin/activate ]; then . /venv/main/bin/activate; fi
nvidia-smi || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'available', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 'torch cannot use this host GPU - driver too old for this image, see nvidia-smi above')"
: "${PULSAR_AGENT_REF:?PULSAR_AGENT_REF must name the commit of the agent code}"
mkdir -p /opt/apps/pulsar
curl -fsSL --retry 3 --retry-delay 5 -o /tmp/pulsar-agent.tar.gz "https://codeload.github.com/Daratic/pulsar-provisioning/tar.gz/${PULSAR_AGENT_REF}"
tar -xzf /tmp/pulsar-agent.tar.gz -C /opt/apps/pulsar --strip-components=1 --wildcards '*/agent/*'
test -f /opt/apps/pulsar/agent/agent_server_multi.py
pip install fastapi==0.115.0 'uvicorn[standard]==0.32.0' pydantic==2.9.2 soundfile torchcodec huggingface_hub
git clone --depth 1 https://github.com/ace-step/ACE-Step.git /opt/ACE-Step || (sleep 5; rm -rf /opt/ACE-Step; git clone --depth 1 https://github.com/ace-step/ACE-Step.git /opt/ACE-Step) || (sleep 15; rm -rf /opt/ACE-Step; git clone --depth 1 https://github.com/ace-step/ACE-Step.git /opt/ACE-Step)
pip install -e /opt/ACE-Step
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/ACE-Step-v1-3.5B', cache_dir='/root/.cache/ace-step/checkpoints')"
export PYTHONPATH=/opt/ACE-Step:/opt/apps/pulsar
python3 -c "import agent.agent_server_multi; print('agent import OK')"
cat > /etc/supervisor/conf.d/pulsar-agent.conf <<'PULSAR_AGENT_CONF'
[program:pulsar-agent]
command=/bin/bash -c 'if [ -f /venv/main/bin/activate ]; then . /venv/main/bin/activate; fi; exec uvicorn agent.agent_server_multi:app --host 0.0.0.0 --port 8188'
directory=/opt/apps/pulsar
environment=PYTHONPATH="/opt/ACE-Step:/opt/apps/pulsar"
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/workspace/agent.log
stdout_logfile_maxbytes=0
redirect_stderr=true
PULSAR_AGENT_CONF
supervisorctl reread
supervisorctl update
touch /workspace/.pulsar_ready
