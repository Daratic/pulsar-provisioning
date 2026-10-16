"""Stage 4: re-voice the separated vocal from a reference clip."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess  # noqa: F401 - run_tool's Popen; tests patch it through here
import uuid

from agent.runners.cover_engines import AUDIO_TOOLS_APT, EngineInstall, StageLog, VoiceEngine
from agent.runners.cover_engines._proc import run_tool

# Pinned to the HEAD of each repo's default branch on 2026-10-16
# (api.github.com/repos/<owner>/<repo>/commits/main).
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"  # 2025-04-20
SEED_VC_ROOT = "/opt/seed-vc"
YINGMUSIC_REPO = "https://github.com/GiantAILab/YingMusic-SVC.git"
YINGMUSIC_COMMIT = "4974a80c6044c4557059548409379f6365129f88"  # 2025-12-29
YINGMUSIC_ROOT = "/opt/yingmusic"
YINGMUSIC_CHECKPOINT = "YingMusic-SVC-full.pt"

# Both repos' requirements.txt pin torch twice: three nightly cu126 lines
# ("torch --pre --index-url ...") and torch==2.4.0/torchvision==0.19.0/
# torchaudio==2.4.0. Installing either would replace the image's torch. This
# matches all six and nothing else (not rotary_embedding_torch, not
# torchcrepe). Valid for both `grep -viE` and Python's re (IGNORECASE).
STRIP_TORCH_PATTERN = r"^(torch|torchaudio|torchvision)([ =<>~!]|$)"

# Seed-VC and YingMusic-SVC run in their OWN Python 3.10 venv, apart from
# the image's Python that runs the agent, audio-separator and MusicGen. They
# have to: both pin numpy==1.26.4 while audio-separator 0.47.0 requires
# numpy>=2, and YingMusic calls torchaudio.sox_effects, which later
# torchaudio releases removed. Both repos were tested on torch 2.4.0, so the
# venv gets that build from the cu124 index (the CUDA runtime ships inside
# the wheels; the host's newer driver runs it). uv supplies CPython 3.10, as
# it does for ddsp_timbre. Both voice engines list VOICE_VENV_SETUP; the
# catalogue runs each distinct install line once.
VOICE_VENV = "/opt/voice/venv"
VOICE_PYTHON = f"{VOICE_VENV}/bin/python"
VOICE_UV_VERSION = "0.12.20"
VOICE_TORCH = ("torch==2.4.0", "torchaudio==2.4.0")
VOICE_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
VOICE_VENV_SETUP = (
    f"pip install uv=={VOICE_UV_VERSION}",
    # --clear: the provisioner reruns the whole script after a failure, and
    # uv refuses to create a venv over an existing one.
    f"uv venv --clear --python 3.10 {VOICE_VENV}",
    f"uv pip install --python {VOICE_PYTHON} pip setuptools wheel",
    # --no-cache-dir: about 2.5 GB of wheels that would otherwise stay in
    # pip's cache on a 50 GB disk.
    f"{VOICE_PYTHON} -m pip install --no-cache-dir {' '.join(VOICE_TORCH)} --index-url {VOICE_TORCH_INDEX}",
)

SEMITONE_RANGE = (-24, 24)
DIFFUSION_STEPS_RANGE = (1, 200)
DEFAULT_DIFFUSION_STEPS = 30
# Model load plus conversion of a whole song; half an hour means it is stuck.
DEFAULT_VOICE_TIMEOUT_SECONDS = 1800

# BigVGAN and whisper are loaded with from_pretrained at runtime; fetch only
# what that reads (the discriminator/optimizer state is not needed).
_BIGVGAN = "snapshot_download('nvidia/bigvgan_v2_44khz_128band_512x', allow_patterns=['*.json', 'bigvgan_generator.pt'])"
_WHISPER = "snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.safetensors', '*.txt'])"


def pinned_clone(url: str, dest: str, commit: str) -> tuple[str, ...]:
    """Fetch exactly `commit` into `dest` (shallow), retried like git_clone.

    One command per line, never an && chain: under `set -e` a failure in a
    non-final position of one does not stop the script.
    """
    fetch = f"git -C {dest} fetch --depth 1 origin {commit}"
    return (
        f"mkdir -p {dest}",
        f"git -C {dest} rev-parse --git-dir >/dev/null 2>&1 || git init -q {dest}",
        f"git -C {dest} remote add origin {url} 2>/dev/null || git -C {dest} remote set-url origin {url}",
        f"{fetch} || (sleep 10; {fetch}) || (sleep 30; {fetch})",
        f"git -C {dest} checkout -q --force FETCH_HEAD",
    )


def install_requirements_without_torch(root: str) -> tuple[str, ...]:
    """The repo's requirements minus every torch line, into the voice venv.

    The torch pins are repeated so pip cannot move the venv's torch while
    resolving the rest; 2.4.0+cu124 satisfies ==2.4.0, so they are a no-op.
    """
    return (
        f"grep -viE '{STRIP_TORCH_PATTERN}' {root}/requirements.txt > {root}/requirements.notorch.txt",
        f"{VOICE_PYTHON} -m pip install --no-cache-dir -r {root}/requirements.notorch.txt {' '.join(VOICE_TORCH)}",
    )


def _voice_predownload(root: str, *downloads: str) -> str:
    """One command that fetches into `root`'s checkpoints, where the scripts look.

    hf_utils.load_custom_model_from_hf downloads with cache_dir='./checkpoints'
    relative to the checkout, and inference.py / my_inference.py both set
    HF_HUB_CACHE=./checkpoints/hf_cache before importing transformers, so
    BigVGAN's and whisper's from_pretrained read from there, NOT from
    ~/.cache/huggingface. Absolute paths, so no `cd` is needed.
    """
    return (
        f"HF_HUB_CACHE={root}/checkpoints/hf_cache {VOICE_PYTHON} -c "
        "\"from huggingface_hub import hf_hub_download, snapshot_download; "
        + "; ".join(downloads) + "\""
    )


def _cached(repo: str, filename: str, root: str) -> str:
    return f"hf_hub_download('{repo}', '{filename}', cache_dir='{root}/checkpoints')"


def _int_param(params: dict, key: str, default: int, bounds: tuple[int, int]) -> int:
    raw = params.get(key, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{key} must be a whole number, got {raw!r}")
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}, got {value}")
    return value


def _single_wav(directory: str, pattern: str, tool: str) -> str:
    found = sorted(glob.glob(os.path.join(directory, pattern)))
    if len(found) != 1:
        raise RuntimeError(f"{tool} should write one {pattern} in {directory}, found {[os.path.basename(f) for f in found]}")
    return found[0]


class SeedVcVoice(VoiceEngine):
    """Seed-VC singing mode (44.1 kHz, f0-conditioned). Zero-shot: any voice
    from a 10-30 s reference clip. Code GPL-3.0."""

    INSTALL = EngineInstall(
        setup_commands=(
            *VOICE_VENV_SETUP,
            *pinned_clone(SEED_VC_REPO, SEED_VC_ROOT, SEED_VC_COMMIT),
            *install_requirements_without_torch(SEED_VC_ROOT),
        ),
        predownload_commands=(
            # Exactly what inference.py loads for --f0-condition True with no
            # --checkpoint: the 44k singing DiT and its config, rmvpe,
            # campplus, then BigVGAN and whisper-small (the config's names).
            _voice_predownload(
                SEED_VC_ROOT,
                _cached("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                        SEED_VC_ROOT),
                _cached("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml", SEED_VC_ROOT),
                _cached("lj1995/VoiceConversionWebUI", "rmvpe.pt", SEED_VC_ROOT),
                _cached("funasr/campplus", "campplus_cn_common.bin", SEED_VC_ROOT),
                _BIGVGAN,
                _WHISPER,
            ),
        ),
    )

    def __init__(self, root: str = SEED_VC_ROOT,
                 stage_timeout_seconds: float = DEFAULT_VOICE_TIMEOUT_SECONDS):
        self._root = root
        self._timeout = stage_timeout_seconds

    def run(self, vocals: str, reference: str | None, params: dict, workdir: str,
            log: StageLog) -> tuple[str, bool]:
        if not reference:
            raise ValueError("seed_vc needs a reference clip (reference_path)")
        shift = _int_param(params, "semitone_shift", 0, SEMITONE_RANGE)
        steps = _int_param(params, "diffusion_steps", DEFAULT_DIFFUSION_STEPS, DIFFUSION_STEPS_RANGE)
        out_dir = os.path.join(workdir, "voice")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        run_tool(
            [
                VOICE_PYTHON, "inference.py",
                "--source", vocals,
                "--target", reference,
                "--output", out_dir,
                "--diffusion-steps", str(steps),
                "--f0-condition", "True",
                "--auto-f0-adjust", "False",
                "--semi-tone-shift", str(shift),
                "--fp16", "True",
            ],
            log,
            "seed-vc inference.py",
            self._timeout,
            cwd=self._root,
        )
        # Named vc_{source}_{target}_{length_adjust}_{steps}_{cfg}.wav.
        produced = _single_wav(out_dir, "vc_*.wav", "seed-vc")
        final = os.path.join(workdir, "vocals_converted.wav")
        shutil.copyfile(produced, final)
        return final, True


class YingMusicSvcVoice(VoiceEngine):
    """YingMusic-SVC (GiantAILab): beats Seed-VC on singing in its paper and
    tolerates separation residue. Code MIT, weights CC-BY-NC-4.0.

    Its script has NO pitch-shift flag (pitch is adapted automatically) and
    NO output-dir flag: it writes ./outputs/<expname>/<tgt>_<src>_None.wav
    relative to its cwd. So `semitone_shift` is IGNORED by this engine, and
    each job uses a fresh expname to find its own file.
    """

    INSTALL = EngineInstall(
        setup_commands=(
            # sox: Remix/auger.py calls torchaudio.sox_effects.
            *AUDIO_TOOLS_APT,
            *VOICE_VENV_SETUP,
            *pinned_clone(YINGMUSIC_REPO, YINGMUSIC_ROOT, YINGMUSIC_COMMIT),
            *install_requirements_without_torch(YINGMUSIC_ROOT),
        ),
        predownload_commands=(
            # The runner passes --checkpoint ./checkpoints/<file> from the
            # checkout, hence local_dir. The rest as for Seed-VC.
            _voice_predownload(
                YINGMUSIC_ROOT,
                f"hf_hub_download('GiantAILab/YingMusic-SVC', '{YINGMUSIC_CHECKPOINT}', "
                f"local_dir='{YINGMUSIC_ROOT}/checkpoints')",
                _cached("lj1995/VoiceConversionWebUI", "rmvpe.pt", YINGMUSIC_ROOT),
                _cached("funasr/campplus", "campplus_cn_common.bin", YINGMUSIC_ROOT),
                _BIGVGAN,
                _WHISPER,
            ),
        ),
    )

    def __init__(self, root: str = YINGMUSIC_ROOT,
                 stage_timeout_seconds: float = DEFAULT_VOICE_TIMEOUT_SECONDS):
        self._root = root
        self._timeout = stage_timeout_seconds

    def run(self, vocals: str, reference: str | None, params: dict, workdir: str,
            log: StageLog) -> tuple[str, bool]:
        if not reference:
            raise ValueError("yingmusic_svc needs a reference clip (reference_path)")
        steps = _int_param(params, "diffusion_steps", DEFAULT_DIFFUSION_STEPS, DIFFUSION_STEPS_RANGE)
        shift = params.get("semitone_shift")
        if shift not in (None, "", 0, "0"):
            log(f"semitone_shift is ignored by yingmusic_svc (got {shift!r}): its pitch adaptation is automatic")
        expname = f"pulsar_{uuid.uuid4().hex[:12]}"
        exp_dir = os.path.join(self._root, "outputs", expname)
        try:
            run_tool(
                [
                    VOICE_PYTHON, "my_inference.py",
                    "--source", vocals,
                    "--target", reference,
                    "--diffusion-steps", str(steps),
                    "--checkpoint", f"./checkpoints/{YINGMUSIC_CHECKPOINT}",
                    "--expname", expname,
                    "--cuda", "0",
                    "--fp16", "True",
                    "--config", "./configs/YingMusic-SVC.yml",
                ],
                log,
                "yingmusic my_inference.py",
                self._timeout,
                cwd=self._root,
            )
            produced = _single_wav(exp_dir, "*.wav", "yingmusic")
            final = os.path.join(workdir, "vocals_converted.wav")
            shutil.copyfile(produced, final)
        finally:
            shutil.rmtree(exp_dir, ignore_errors=True)
        return final, True


class KeepOriginalVoice(VoiceEngine):
    """No conversion: the separated vocal goes to the mix as it is."""

    def run(self, vocals: str, reference: str | None, params: dict, workdir: str,
            log: StageLog) -> tuple[str, bool]:
        return vocals, False
