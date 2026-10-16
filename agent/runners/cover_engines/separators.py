"""Stage 1: split the source mix into vocals and instrumental."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess  # noqa: F401 - run_tool's Popen; tests patch it through here

from agent.runners.cover_engines import AUDIO_TOOLS_APT, EngineInstall, Separator, StageLog
from agent.runners.cover_engines._proc import run_tool

# BS-RoFormer (ep 317, SDR 12.98), via the audio-separator CLI. Better vocal
# separation than htdemucs_ft and the separator YingMusic-SVC itself ships.
BS_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# Under /opt, not the CLI's /tmp default, so a parked box keeps the weights.
BS_ROFORMER_MODEL_DIR = "/opt/audio-separator-models"
# Installed with --no-deps, and its runtime dependencies listed below
# instead. Its own list wants diffq (used only by the Demucs-architecture
# models, which we never load), and diffq 0.2.4 has no wheel past Python
# 3.10, so on the image's Python pip would try to compile it. The [gpu]
# extra adds only onnxruntime-gpu, listed here. torch stays the image's own:
# audio-separator asks for torch>=2.3, which it already satisfies.
AUDIO_SEPARATOR_PIN = "audio-separator==0.47.0"
AUDIO_SEPARATOR_DEPS = (
    "onnxruntime-gpu>=1.17",  # separator.py imports onnxruntime at module level
    "beartype>=0.18.5,<0.19.0",
    "einops>=0.7",
    "rotary-embedding-torch>=0.6.1,<0.7.0",
    "ml_collections",
    "julius>=0.2",
    "pydub>=0.25",
    "pyyaml",
    "requests>=2",
    "six>=1.16",
    "tqdm",
    "packaging",
    "scipy>=1.13.0,<2.0.0",
    "librosa==0.11.0",
    "soundfile>=0.12",
)
CUSTOM_OUTPUT_NAMES = {"Vocals": "vocals", "Instrumental": "instrumental"}

DEMUCS_MODEL = "htdemucs_ft"

# A separation of a long song takes minutes; twenty means the tool is stuck.
DEFAULT_SEPARATOR_TIMEOUT_SECONDS = 1200


class BsRoformerSeparator(Separator):
    INSTALL = EngineInstall(
        pip_packages=AUDIO_SEPARATOR_DEPS,
        # ffmpeg: compressed sources are decoded through librosa/pydub.
        setup_commands=(*AUDIO_TOOLS_APT, f"pip install --no-deps {AUDIO_SEPARATOR_PIN}"),
        predownload_commands=(
            f"audio-separator --download_model_only --model_filename {BS_ROFORMER_MODEL} "
            f"--model_file_dir {BS_ROFORMER_MODEL_DIR}",
        ),
    )

    def __init__(self, stage_timeout_seconds: float = DEFAULT_SEPARATOR_TIMEOUT_SECONDS):
        self._timeout = stage_timeout_seconds

    def run(self, src: str, workdir: str, log: StageLog) -> tuple[str, str]:
        out_dir = os.path.join(workdir, "sep")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        run_tool(
            [
                "audio-separator", src,
                "--model_filename", BS_ROFORMER_MODEL,
                "--model_file_dir", BS_ROFORMER_MODEL_DIR,
                "--output_dir", out_dir,
                "--output_format", "WAV",
                # One argv element, no shell: the JSON needs no quoting.
                "--custom_output_names", json.dumps(CUSTOM_OUTPUT_NAMES),
            ],
            log,
            "audio-separator",
            self._timeout,
        )
        return locate_stems(out_dir)


def locate_stems(out_dir: str) -> tuple[str, str]:
    """(vocals, instrumental) among the WAVs in out_dir.

    Tolerant of naming: with --custom_output_names they are vocals.wav and
    instrumental.wav; without, `<base>_(Vocals)_<model>.wav` and
    `<base>_(Instrumental)_<model>.wav`. "no_vocals" counts as instrumental.
    """
    files = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
    instrumental = [f for f in files if _is_instrumental(os.path.basename(f))]
    vocals = [f for f in files if "vocals" in os.path.basename(f).lower() and f not in instrumental]
    if len(vocals) != 1 or len(instrumental) != 1:
        names = [os.path.basename(f) for f in files]
        raise RuntimeError(f"expected exactly two stems (vocals, instrumental) in {out_dir}, found {names}")
    return vocals[0], instrumental[0]


def _is_instrumental(name: str) -> bool:
    lowered = name.lower()
    return "instrumental" in lowered or "no_vocals" in lowered


class DemucsSeparator(Separator):
    """The fallback: htdemucs_ft two-stem, the existing demucs runner's model."""

    INSTALL = EngineInstall(
        pip_packages=("demucs==4.1.0",),
        predownload_commands=(
            f"python3 -c \"from demucs import pretrained; pretrained.get_model('{DEMUCS_MODEL}')\"",
        ),
    )

    def __init__(self, stage_timeout_seconds: float = DEFAULT_SEPARATOR_TIMEOUT_SECONDS):
        self._timeout = stage_timeout_seconds

    def run(self, src: str, workdir: str, log: StageLog) -> tuple[str, str]:
        out_dir = os.path.join(workdir, "sep")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        run_tool(
            ["python3", "-m", "demucs", "-n", DEMUCS_MODEL, "-o", out_dir, "--two-stems", "vocals", src],
            log,
            "demucs",
            self._timeout,
        )
        track = os.path.splitext(os.path.basename(src))[0]
        stem_dir = os.path.join(out_dir, DEMUCS_MODEL, track)
        vocals = os.path.join(stem_dir, "vocals.wav")
        instrumental = os.path.join(stem_dir, "no_vocals.wav")
        missing = [p for p in (vocals, instrumental) if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"demucs exited 0 but wrote no stems at {missing}")
        return vocals, instrumental
