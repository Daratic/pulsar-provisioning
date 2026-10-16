"""Small audio helpers shared by the backing engine and the mixer.

numpy/soundfile are imported inside the functions: the agent imports this
module on boxes where the cover engines are not installed.
"""

from __future__ import annotations

from math import gcd


def read_audio(path: str):
    """(samples, rate) with samples shaped (frames, channels), float32."""
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return data, rate


def to_mono(data):
    return data.mean(axis=1) if data.ndim == 2 else data


def to_stereo(data):
    import numpy as np

    if data.ndim == 1:
        return np.stack([data, data], axis=1)
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def resample(data, rate_from: int, rate_to: int):
    """Resample along axis 0. scipy's polyphase resampler; librosa if scipy
    is missing (both are on the box - librosa depends on scipy)."""
    if rate_from == rate_to:
        return data
    try:
        from scipy.signal import resample_poly

        divisor = gcd(int(rate_from), int(rate_to))
        return resample_poly(data, int(rate_to) // divisor, int(rate_from) // divisor, axis=0).astype("float32")
    except ImportError:
        import librosa

        return librosa.resample(data, orig_sr=rate_from, target_sr=rate_to, axis=0).astype("float32")


def fit_length(data, frames: int):
    """Trim or zero-pad along axis 0 to exactly `frames`."""
    import numpy as np

    if len(data) >= frames:
        return data[:frames]
    pad = [(0, frames - len(data))] + [(0, 0)] * (data.ndim - 1)
    return np.pad(data, pad)


def audio_duration(path: str) -> float:
    import soundfile as sf

    return float(sf.info(path).duration)


def fit_file_to_seconds(path: str, seconds: float) -> None:
    """Rewrite a WAV in place, trimmed or zero-padded to `seconds`, keeping
    its rate, channels and sample format."""
    import soundfile as sf

    info = sf.info(path)
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    sf.write(path, fit_length(data, int(round(seconds * rate))), rate, subtype=info.subtype)
