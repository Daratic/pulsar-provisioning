#!/bin/bash
# Generated from Pulsar's app/model_adapters/runner_catalog.py by
# scripts/generate_provisioning.py. Do not edit by hand: change the
# catalog, regenerate, export, and move Pulsar's pin.
# Runner set: cover_pipeline
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
pip install fastapi==0.115.0 'uvicorn[standard]==0.32.0' pydantic==2.9.2 soundfile librosa==0.11.0 scipy 'onnxruntime-gpu>=1.17' 'beartype>=0.18.5,<0.19.0' 'einops>=0.7' 'rotary-embedding-torch>=0.6.1,<0.7.0' ml_collections 'julius>=0.2' 'pydub>=0.25' pyyaml 'requests>=2' 'six>=1.16' tqdm packaging 'scipy>=1.13.0,<2.0.0' 'soundfile>=0.12' demucs==4.1.0 'transformers>=4.46,<4.58' accelerate huggingface_hub pyloudnorm==0.2.0
apt-get update
apt-get install -y --no-install-recommends sox libsox-fmt-all ffmpeg
pip install --no-deps audio-separator==0.47.0
pip install uv==0.12.20
uv venv --clear --python 3.10 /opt/voice/venv
uv pip install --python /opt/voice/venv/bin/python pip setuptools wheel
/opt/voice/venv/bin/python -m pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
mkdir -p /opt/seed-vc
git -C /opt/seed-vc rev-parse --git-dir >/dev/null 2>&1 || git init -q /opt/seed-vc
git -C /opt/seed-vc remote add origin https://github.com/Plachtaa/seed-vc.git 2>/dev/null || git -C /opt/seed-vc remote set-url origin https://github.com/Plachtaa/seed-vc.git
git -C /opt/seed-vc fetch --depth 1 origin 51383efd921027683c89e5348211d93ff12ac2a8 || (sleep 10; git -C /opt/seed-vc fetch --depth 1 origin 51383efd921027683c89e5348211d93ff12ac2a8) || (sleep 30; git -C /opt/seed-vc fetch --depth 1 origin 51383efd921027683c89e5348211d93ff12ac2a8)
git -C /opt/seed-vc checkout -q --force FETCH_HEAD
grep -viE '^(torch|torchaudio|torchvision)([ =<>~!]|$)' /opt/seed-vc/requirements.txt > /opt/seed-vc/requirements.notorch.txt
/opt/voice/venv/bin/python -m pip install --no-cache-dir -r /opt/seed-vc/requirements.notorch.txt torch==2.4.0 torchaudio==2.4.0
mkdir -p /opt/yingmusic
git -C /opt/yingmusic rev-parse --git-dir >/dev/null 2>&1 || git init -q /opt/yingmusic
git -C /opt/yingmusic remote add origin https://github.com/GiantAILab/YingMusic-SVC.git 2>/dev/null || git -C /opt/yingmusic remote set-url origin https://github.com/GiantAILab/YingMusic-SVC.git
git -C /opt/yingmusic fetch --depth 1 origin 4974a80c6044c4557059548409379f6365129f88 || (sleep 10; git -C /opt/yingmusic fetch --depth 1 origin 4974a80c6044c4557059548409379f6365129f88) || (sleep 30; git -C /opt/yingmusic fetch --depth 1 origin 4974a80c6044c4557059548409379f6365129f88)
git -C /opt/yingmusic checkout -q --force FETCH_HEAD
grep -viE '^(torch|torchaudio|torchvision)([ =<>~!]|$)' /opt/yingmusic/requirements.txt > /opt/yingmusic/requirements.notorch.txt
/opt/voice/venv/bin/python -m pip install --no-cache-dir -r /opt/yingmusic/requirements.notorch.txt torch==2.4.0 torchaudio==2.4.0
python3 -c "import audio_separator, pyloudnorm, librosa, demucs; from audio_separator.separator import Separator; from audio_separator.separator.architectures import mdxc_separator; from audio_separator.separator.uvr_lib_v5.roformer import bs_roformer; from transformers import AutoTokenizer, MusicgenMelodyForConditionalGeneration; print('cover main python OK')"
/opt/voice/venv/bin/python -c "import torch, torchaudio; print('voice venv torch', torch.__version__, 'cuda', torch.cuda.is_available())"
/opt/voice/venv/bin/python -c "import yaml, hydra, transformers, librosa, soundfile; print('voice venv OK')"
PYTHONPATH=/opt/apps/pulsar python3 -c "import numpy; from agent.runners.cover_engines.backing import compute_chroma_features; print('backing chroma', compute_chroma_features(numpy.zeros(32000, dtype='float32')).shape)"
PYTHONPATH=/opt/apps/pulsar python3 -c "import agent.runners.cover_pipeline; print('cover_pipeline import OK')"
audio-separator --download_model_only --model_filename model_bs_roformer_ep_317_sdr_12.9755.ckpt --model_file_dir /opt/audio-separator-models || (sleep 10; audio-separator --download_model_only --model_filename model_bs_roformer_ep_317_sdr_12.9755.ckpt --model_file_dir /opt/audio-separator-models) || (sleep 30; audio-separator --download_model_only --model_filename model_bs_roformer_ep_317_sdr_12.9755.ckpt --model_file_dir /opt/audio-separator-models)
python3 -c "from demucs import pretrained; pretrained.get_model('htdemucs_ft')" || (sleep 10; python3 -c "from demucs import pretrained; pretrained.get_model('htdemucs_ft')") || (sleep 30; python3 -c "from demucs import pretrained; pretrained.get_model('htdemucs_ft')")
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/musicgen-melody', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*', '*.yaml'])" || (sleep 10; python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/musicgen-melody', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*', '*.yaml'])") || (sleep 30; python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/musicgen-melody', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*', '*.yaml'])")
HF_HUB_CACHE=/opt/seed-vc/checkpoints/hf_cache /opt/voice/venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; hf_hub_download('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('Plachta/Seed-VC', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('funasr/campplus', 'campplus_cn_common.bin', cache_dir='/opt/seed-vc/checkpoints'); snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt']); snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])" || (sleep 10; HF_HUB_CACHE=/opt/seed-vc/checkpoints/hf_cache /opt/voice/venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; hf_hub_download('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('Plachta/Seed-VC', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('funasr/campplus', 'campplus_cn_common.bin', cache_dir='/opt/seed-vc/checkpoints'); snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt']); snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])") || (sleep 30; HF_HUB_CACHE=/opt/seed-vc/checkpoints/hf_cache /opt/voice/venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; hf_hub_download('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('Plachta/Seed-VC', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/opt/seed-vc/checkpoints'); hf_hub_download('funasr/campplus', 'campplus_cn_common.bin', cache_dir='/opt/seed-vc/checkpoints'); snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt']); snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])")
HF_HUB_CACHE=/opt/yingmusic/checkpoints/hf_cache /opt/voice/venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; hf_hub_download('GiantAILab/YingMusic-SVC', 'YingMusic-SVC-full.pt', local_dir='/opt/yingmusic/checkpoints'); hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/opt/yingmusic/checkpoints'); hf_hub_download('funasr/campplus', 'campplus_cn_common.bin', cache_dir='/opt/yingmusic/checkpoints'); snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt']); snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])" || (sleep 10; HF_HUB_CACHE=/opt/yingmusic/checkpoints/hf_cache /opt/voice/venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; hf_hub_download('GiantAILab/YingMusic-SVC', 'YingMusic-SVC-full.pt', local_dir='/opt/yingmusic/checkpoints'); hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/opt/yingmusic/checkpoints'); hf_hub_download('funasr/campplus', 'campplus_cn_common.bin', cache_dir='/opt/yingmusic/checkpoints'); snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt']); snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])") || (sleep 30; HF_HUB_CACHE=/opt/yingmusic/checkpoints/hf_cache /opt/voice/venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; hf_hub_download('GiantAILab/YingMusic-SVC', 'YingMusic-SVC-full.pt', local_dir='/opt/yingmusic/checkpoints'); hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/opt/yingmusic/checkpoints'); hf_hub_download('funasr/campplus', 'campplus_cn_common.bin', cache_dir='/opt/yingmusic/checkpoints'); snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt']); snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])")
export PYTHONPATH=/opt/apps/pulsar
python3 -c "import agent.agent_server_multi; print('agent import OK')"
cat > /etc/supervisor/conf.d/pulsar-agent.conf <<'PULSAR_AGENT_CONF'
[program:pulsar-agent]
command=/bin/bash -c 'if [ -f /venv/main/bin/activate ]; then . /venv/main/bin/activate; fi; exec uvicorn agent.agent_server_multi:app --host 0.0.0.0 --port 8188'
directory=/opt/apps/pulsar
environment=PYTHONPATH="/opt/apps/pulsar"
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
