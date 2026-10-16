"""bpm, key and time signature of a source track, for ACE-Step's anchors.

ACE-Step's tutorial: the model treats bpm, key and time signature as anchors
it follows loosely, and recommends setting them when matching existing
material - which a cover always is. Runs on the rented instance, where
librosa is installed; imported lazily so Pulsar's own tests need no librosa.

- bpm: librosa.beat.beat_track.
- key: Krumhansl-Schmuckler - correlate the track's mean chroma (chroma_cqt)
  with the 24 rotated Krumhansl-Kessler (1982) key profiles; the best
  correlation wins.
- time signature: librosa has no reliable metre estimator, so it is 4/4,
  sent in ACE-Step's own encoding, "4" (VALID_TIME_SIGNATURES = [2, 3, 4, 6]).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_TIME_SIGNATURE = "4"
# ACE-Step's own BPM range (acestep/constants.py).
BPM_MIN = 30
BPM_MAX = 300
ANALYSIS_SAMPLE_RATE = 22050
# Enough to find tempo and key; bounds the CPU time on a billing rental.
ANALYSIS_MAX_SECONDS = 300.0

# Krumhansl-Kessler profiles rotated to every tonic. Index 0 is C, 1 C#, ...
# 11 B. Names are spelt as ACE-Step's VALID_KEYSCALES accepts them.
KEY_PROFILES: dict[str, tuple[float, ...]] = {
    "C major": (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
    "C# major": (2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29),
    "D major": (2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66),
    "D# major": (3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39),
    "E major": (2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19),
    "F major": (5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52),
    "F# major": (2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09),
    "G major": (4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38),
    "G# major": (4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33),
    "A major": (2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48),
    "A# major": (3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23),
    "B major": (2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35),
    "C minor": (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    "C# minor": (3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34),
    "D minor": (3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69),
    "D# minor": (2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98),
    "E minor": (3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75),
    "F minor": (4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54),
    "F# minor": (2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53),
    "G minor": (3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60),
    "G# minor": (2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38),
    "A minor": (5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52),
    "A# minor": (3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68),
    "B minor": (2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33),
}


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    spread_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    spread_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if spread_a == 0 or spread_b == 0:
        return 0.0
    return covariance / (spread_a * spread_b)


def estimate_key(chroma_profile: Sequence[float]) -> str:
    values = [float(v) for v in chroma_profile]
    if len(values) != 12:
        raise ValueError(f"a chroma profile has 12 pitch classes, got {len(values)}")
    return max(KEY_PROFILES, key=lambda name: _pearson(values, KEY_PROFILES[name]))


def bpm_from_tempo(tempo) -> int:
    # librosa 0.11 returns tempo as a one-element array; older ones, a float.
    try:
        value = float(tempo)
    except TypeError:
        value = float(tempo[0])
    return int(min(BPM_MAX, max(BPM_MIN, round(value))))


def estimate_metas(path: str, librosa_module=None) -> dict:
    librosa = librosa_module
    if librosa is None:
        import librosa  # only installed on the rented instance
    y, sr = librosa.load(path, sr=ANALYSIS_SAMPLE_RATE, mono=True, duration=ANALYSIS_MAX_SECONDS)
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = [float(v) for v in chroma.mean(axis=1)]
    return {
        "bpm": bpm_from_tempo(tempo),
        "keyscale": estimate_key(profile),
        "timesignature": DEFAULT_TIME_SIGNATURE,
    }
