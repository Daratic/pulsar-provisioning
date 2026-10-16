#!/bin/bash
# Generated from Pulsar's app/model_adapters/runner_catalog.py by
# scripts/generate_provisioning.py. Do not edit by hand: change the
# catalog, regenerate, export, and move Pulsar's pin.
# Runner set: ace_step_15
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
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print('torch now', torch.__version__, 'cuda', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 'CUDA lost after torch swap')"
pip install fastapi==0.115.0 'uvicorn[standard]==0.32.0' pydantic==2.9.2 soundfile torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 'torchcodec==0.10.*' huggingface_hub 'transformers>=4.51.0,<4.58.0' 'diffusers>=0.37.0' 'accelerate>=1.12.0' 'vector-quantize-pytorch>=1.27.15' 'torchao>=0.16.0,<0.17.0' 'pytorch-wavelets>=1.3.0' 'pywavelets>=1.9.0' 'numba>=0.63.1' 'typer-slim>=0.21.1' diskcache 'loguru>=0.7.3' 'einops>=0.8.1' 'scipy>=1.10.1' toml librosa==0.11.0 'xxhash>=3.5.0'
git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5 || (sleep 5; rm -rf /opt/ACE-Step-1.5; git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5) || (sleep 15; rm -rf /opt/ACE-Step-1.5; git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5)
git -C /opt/ACE-Step-1.5 fetch --depth 1 origin 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0 || (sleep 5; git -C /opt/ACE-Step-1.5 fetch --depth 1 origin 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0) || (sleep 15; git -C /opt/ACE-Step-1.5 fetch --depth 1 origin 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0)
git -C /opt/ACE-Step-1.5 checkout --detach 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
pip install fastapi==0.115.0 'uvicorn[standard]==0.32.0' pydantic==2.9.2 soundfile
python3 -c "import fastapi, uvicorn, pydantic, soundfile; print('agent deps OK')"
pip install --no-deps /opt/ACE-Step-1.5/acestep/third_parts/nano-vllm
pip install --no-deps -e /opt/ACE-Step-1.5
python3 -c "import torchaudio; import torchcodec; from torchcodec.decoders import AudioDecoder; from acestep.handler import AceStepHandler; from acestep.llm_inference import LLMHandler; from acestep.inference import GenerationParams, GenerationConfig, generate_music; print('acestep import OK')"
python3 -c "import sys; from acestep.model_downloader import ensure_main_model; ok, msg = ensure_main_model('/opt/ACE-Step-1.5/checkpoints', prefer_source='huggingface'); print(msg); sys.exit(0 if ok else 1)" || (sleep 10; python3 -c "import sys; from acestep.model_downloader import ensure_main_model; ok, msg = ensure_main_model('/opt/ACE-Step-1.5/checkpoints', prefer_source='huggingface'); print(msg); sys.exit(0 if ok else 1)") || (sleep 30; python3 -c "import sys; from acestep.model_downloader import ensure_main_model; ok, msg = ensure_main_model('/opt/ACE-Step-1.5/checkpoints', prefer_source='huggingface'); print(msg); sys.exit(0 if ok else 1)")
python3 -c "import sys; from acestep.model_downloader import check_model_exists, download_submodel; ok, msg = (True, 'DiT present') if check_model_exists('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints') else download_submodel('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints', force=True, prefer_source='huggingface'); print(msg); sys.exit(0 if ok else 1)" || (sleep 10; python3 -c "import sys; from acestep.model_downloader import check_model_exists, download_submodel; ok, msg = (True, 'DiT present') if check_model_exists('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints') else download_submodel('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints', force=True, prefer_source='huggingface'); print(msg); sys.exit(0 if ok else 1)") || (sleep 30; python3 -c "import sys; from acestep.model_downloader import check_model_exists, download_submodel; ok, msg = (True, 'DiT present') if check_model_exists('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints') else download_submodel('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints', force=True, prefer_source='huggingface'); print(msg); sys.exit(0 if ok else 1)")
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='/root/.cache/ace-step/checkpoints/acestep-5Hz-lm-0.6B')" || (sleep 10; python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='/root/.cache/ace-step/checkpoints/acestep-5Hz-lm-0.6B')") || (sleep 30; python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='/root/.cache/ace-step/checkpoints/acestep-5Hz-lm-0.6B')")
python3 -c "import sys; from acestep.model_downloader import check_main_model_exists, check_model_exists, check_vae_exists; missing = [n for n, ok in (('main model', check_main_model_exists('/opt/ACE-Step-1.5/checkpoints')), ('acestep-v15-xl-turbo', check_model_exists('acestep-v15-xl-turbo', '/opt/ACE-Step-1.5/checkpoints')), ('vae', check_vae_exists('official', '/opt/ACE-Step-1.5/checkpoints')), ('acestep-5Hz-lm-0.6B', check_model_exists('acestep-5Hz-lm-0.6B', '/root/.cache/ace-step/checkpoints'))) if not ok]; print('ACE-Step models present' if not missing else 'ACE-Step models MISSING: ' + ', '.join(missing)); sys.exit(1 if missing else 0)"
export PYTHONPATH=/opt/ACE-Step-1.5:/opt/apps/pulsar
python3 -c "import agent.agent_server_multi; print('agent import OK')"
cat > /etc/supervisor/conf.d/pulsar-agent.conf <<'PULSAR_AGENT_CONF'
[program:pulsar-agent]
command=/bin/bash -c 'if [ -f /venv/main/bin/activate ]; then . /venv/main/bin/activate; fi; exec uvicorn agent.agent_server_multi:app --host 0.0.0.0 --port 8188'
directory=/opt/apps/pulsar
environment=PYTHONPATH="/opt/ACE-Step-1.5:/opt/apps/pulsar"
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
