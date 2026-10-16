"""Stage 5: loudness-match the vocal and the backing, sum, limit."""

from __future__ import annotations

import math

from agent.runners.cover_engines import EngineInstall, Mixer, StageLog
from agent.runners.cover_engines.audio import fit_length, read_audio, resample, to_stereo

MIX_RATE = 48000
BACKING_LUFS = -16.0
CEILING_DBTP = -1.0
# True peak is estimated on a 4x oversampled copy (ITU-R BS.1770's method),
# which catches the inter-sample peaks a plain sample peak misses.
TRUE_PEAK_OVERSAMPLE = 4


def _loudness(meter, data) -> float | None:
    """Integrated loudness, or None for silence (pyloudnorm reports -inf)."""
    value = meter.integrated_loudness(data)
    return None if value is None or math.isinf(value) or math.isnan(value) else float(value)


def _gain_to(data, measured: float | None, target: float):
    if measured is None:
        return data
    return data * (10 ** ((target - measured) / 20))


def _true_peak(data) -> float:
    import numpy as np

    try:
        oversampled = resample(data, MIX_RATE, MIX_RATE * TRUE_PEAK_OVERSAMPLE)
        return float(max(np.abs(oversampled).max(initial=0.0), np.abs(data).max(initial=0.0)))
    except Exception:
        return float(np.abs(data).max(initial=0.0))


class LoudnessMixer(Mixer):
    """Deterministic, no model: backing to -16 LUFS, vocal to -16 +
    vocal_level_db, summed, scaled down to a -1 dBTP ceiling if needed,
    written as 48 kHz stereo 24-bit PCM."""

    INSTALL = EngineInstall(pip_packages=("pyloudnorm==0.2.0", "soundfile", "scipy"))

    def run(self, vocals: str, backing: str, vocal_level_db: float, out_path: str, log: StageLog) -> str:
        import numpy as np
        import pyloudnorm as pyln
        import soundfile as sf

        stems = []
        for path in (vocals, backing):
            data, rate = read_audio(path)
            stems.append(to_stereo(resample(data, rate, MIX_RATE)).astype("float64"))
        frames = max(len(s) for s in stems)
        voice, back = (fit_length(s, frames) for s in stems)

        meter = pyln.Meter(MIX_RATE)
        voice_lufs, back_lufs = _loudness(meter, voice), _loudness(meter, back)
        voice_target = BACKING_LUFS + float(vocal_level_db)
        mix = _gain_to(back, back_lufs, BACKING_LUFS) + _gain_to(voice, voice_lufs, voice_target)
        log(f"mix: vocals {voice_lufs} LUFS -> {voice_target:.1f}, backing {back_lufs} LUFS -> {BACKING_LUFS:.1f}")

        ceiling = 10 ** (CEILING_DBTP / 20)
        peak = _true_peak(mix)
        if peak > ceiling:
            mix = mix * (ceiling / peak)
            log(f"mix: true peak {20 * math.log10(peak):.2f} dBTP scaled to {CEILING_DBTP:.1f}")
        # The 24-bit quantiser can round a sample at the ceiling up by one LSB.
        mix = np.clip(mix, -ceiling, ceiling)
        sf.write(out_path, mix, MIX_RATE, subtype="PCM_24")
        return out_path
