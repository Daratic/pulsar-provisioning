"""Stage 3: a new backing on the song's own melody and chords.

MusicGen-Melody (transformers) conditions on the chroma of a melody stem, so
key and harmony follow the source by construction; the text prompt carries
the new style. The model sees at most 30 s at a time (1500 tokens at 50/s;
its chroma window is 30 s), so the song is cut into windows on bar lines,
each generated for exactly its own span of the source - which keeps the
result time-aligned with the vocal - and joined with equal-power crossfades.
"""

from __future__ import annotations

import math
import os
import time

from agent.runners.cover_engines import BackingEngine, EngineInstall, StageLog, collect_and_empty_cache
from agent.runners.cover_engines.audio import fit_length, read_audio, resample, to_mono

SAMPLE_RATE = 32000
TOKENS_PER_SECOND = 50
MAX_NEW_TOKENS = 1500
# What one generate call can produce and condition on.
MAX_GENERATION_SECONDS = 30.0
# Each window is generated this far into the next one, for the crossfade.
OVERLAP_SECONDS = 2.0
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_MODEL = "facebook/musicgen-melody"
DEFAULT_GUIDANCE_SCALE = 3.0

# Safetensors and configs only: the repo's *.bin duplicates would double the
# ~6.5 GB download.
_ALLOW = "['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*', '*.yaml']"


def plan_windows(total_seconds: float, bpm: float, timesig: int, max_window_seconds: float) -> list[tuple[float, float]]:
    """Contiguous (start, end) windows covering [0, total_seconds].

    Every window is a whole number of bars (bar = 60/bpm x beats per bar),
    as many as fit in max_window_seconds; only the last ends at the track's
    end instead of a bar line. If even one bar is longer than the cap, the
    windows are simply the cap long.
    """
    if total_seconds <= 0:
        return []
    bar = 60.0 / float(bpm) * int(timesig)
    bars = math.floor(max_window_seconds / bar + 1e-9)
    window = bars * bar if bars >= 1 else max_window_seconds
    windows = []
    index = 0
    while True:
        start = index * window
        if start >= total_seconds - 1e-9:
            break
        windows.append((start, min(start + window, total_seconds)))
        index += 1
    return windows


def _default_load(model_id: str):  # pragma: no cover - needs torch, transformers and the weights
    import torch
    from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return {
        "processor": processor,
        "model": model,
        "device": device,
        "dtype": dtype,
        "rate": int(model.config.audio_encoder.sampling_rate),
    }


def _default_generate(handle, audio, text, max_new_tokens, guidance_scale):  # pragma: no cover - needs the model
    import torch

    inputs = handle["processor"](
        audio=audio, sampling_rate=SAMPLE_RATE, text=[text], padding=True, return_tensors="pt"
    )
    moved = {
        key: (value.to(handle["device"], dtype=handle["dtype"]) if torch.is_floating_point(value)
              else value.to(handle["device"]))
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        out = handle["model"].generate(
            **moved, do_sample=True, guidance_scale=guidance_scale, max_new_tokens=max_new_tokens
        )
    wave = out[0, 0].float().cpu().numpy()
    del out, moved, inputs
    return resample(wave, handle["rate"], SAMPLE_RATE)


def _default_seed(seed: int) -> None:  # pragma: no cover - needs torch
    import torch

    torch.manual_seed(seed)


class MusicgenMelodyBacking(BackingEngine):
    """Weights CC-BY-NC-4.0: fine for Tom's own music, not commercial output."""

    INSTALL = EngineInstall(
        pip_packages=("transformers>=4.46,<4.58", "accelerate", "huggingface_hub", "soundfile", "scipy"),
        predownload_commands=(
            "python3 -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{DEFAULT_MODEL}', allow_patterns={_ALLOW})\"",
        ),
    )

    def __init__(self, generate_fn=None, load_fn=None, seed_fn=None):
        self._generate_fn = generate_fn or _default_generate
        self._load_fn = load_fn or _default_load
        self._seed_fn = seed_fn or _default_seed

    def plan_spans(self, total_seconds: float, bpm: float, timesig: int,
                   window_seconds: float = DEFAULT_WINDOW_SECONDS) -> list[tuple[float, float, float]]:
        """(start, end, generate_until) per window.

        The bar windows are capped so that a window PLUS its overlap still
        fits one generate call: 30 s windows with a 2 s overlap would ask
        for 32 s, which the model cannot condition on or produce.
        """
        cap = min(float(window_seconds), MAX_GENERATION_SECONDS - OVERLAP_SECONDS)
        windows = plan_windows(total_seconds, bpm, timesig, cap)
        # A tail of OVERLAP_SECONDS or less is never generated on its own: a
        # fraction of a second of chroma and a handful of tokens can fail
        # inside the model after minutes of paid work. The previous window
        # already reaches the end through its overlap, so it simply becomes
        # the last window; at most cap + OVERLAP_SECONDS <= 30 s long.
        if len(windows) > 1 and windows[-1][1] - windows[-1][0] <= OVERLAP_SECONDS + 1e-9:
            tail_start = windows[-2][0]
            windows = windows[:-2] + [(tail_start, total_seconds)]
        spans = []
        for index, (start, end) in enumerate(windows):
            last = index == len(windows) - 1
            spans.append((start, end, end if last else min(end + OVERLAP_SECONDS, total_seconds)))
        return spans

    def run(self, melody_stem: str, caption: str, bpm: float, keyscale: str, timesig: int,
            target_seconds: float, params: dict, workdir: str, log: StageLog) -> str:
        import numpy as np
        import soundfile as sf

        window_seconds = float(params.get("window_seconds") or DEFAULT_WINDOW_SECONDS)
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if window_seconds > MAX_GENERATION_SECONDS:
            log(f"window_seconds {window_seconds} is over the model's {MAX_GENERATION_SECONDS:.0f} s; using 30")
            window_seconds = MAX_GENERATION_SECONDS
        guidance_scale = float(params.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        seed = params.get("seed")
        seed = 0 if seed in (None, "") else int(seed)
        model_id = params.get("musicgen_model") or DEFAULT_MODEL
        text = f"{caption}, {keyscale}, {bpm:.0f} bpm"

        data, rate = read_audio(melody_stem)
        total = int(round(target_seconds * SAMPLE_RATE))
        melody = fit_length(resample(to_mono(data), rate, SAMPLE_RATE), total)
        del data

        spans = self.plan_spans(target_seconds, bpm, timesig, window_seconds)
        log(f"backing: {len(spans)} windows, text={text!r}, model={model_id}, seed={seed}")
        out = np.zeros(total, dtype="float32")
        handle = self._load_fn(model_id)
        try:
            self._seed_fn(seed)
            fade_in = 0
            for index, (start, end, gen_end) in enumerate(spans):
                began = time.monotonic()
                a, b, c = (int(round(x * SAMPLE_RATE)) for x in (start, end, gen_end))
                b, c = min(b, total), min(c, total)
                tokens = min(MAX_NEW_TOKENS, max(1, round(TOKENS_PER_SECOND * (gen_end - start))))
                generated = self._generate_fn(
                    handle, melody[a:c], text, max_new_tokens=tokens, guidance_scale=guidance_scale
                )
                segment = fit_length(np.asarray(generated, dtype="float32").reshape(-1), c - a).copy()
                if fade_in:
                    segment[:fade_in] *= np.sin(_ramp(fade_in) * (math.pi / 2))
                fade_out = c - b
                if fade_out:
                    segment[-fade_out:] *= np.cos(_ramp(fade_out) * (math.pi / 2))
                out[a:c] += segment
                fade_in = fade_out
                log(f"backing window {index + 1}/{len(spans)} {start:.2f}-{gen_end:.2f}s "
                    f"({tokens} tokens) in {time.monotonic() - began:.1f}s")
        finally:
            del handle
            collect_and_empty_cache()

        path = os.path.join(workdir, "backing.wav")
        sf.write(path, np.stack([out, out], axis=1), SAMPLE_RATE)
        return path


def _ramp(n: int):
    """0..1 sample centres, shared by a fade-out and the fade-in it meets, so
    sin^2 + cos^2 = 1 at every sample: equal power, no dip, no bump."""
    import numpy as np

    return ((np.arange(n) + 0.5) / n).astype("float32")
