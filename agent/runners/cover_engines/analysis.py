"""Stage 2: bpm, key and time signature of the instrumental.

A thin wrapper over source_metas.estimate_metas, which already does exactly
this for ACE-Step (librosa beat_track; Krumhansl-Schmuckler: mean chroma_cqt
correlated with the 24 rotated major/minor profiles). One estimator, not two.
"""

from __future__ import annotations

from agent.runners.source_metas import estimate_metas

# When librosa finds no beat at all. 120 is the commonest tempo; the bar grid
# it gives only decides where the backing's windows are cut.
DEFAULT_BPM = 120


def measure_bpm_key(path: str, librosa_module=None) -> tuple[float, str, int]:
    """(bpm, keyscale, timesig) - timesig as beats per bar (always 4: librosa
    has no reliable metre estimator)."""
    metas = estimate_metas(path, librosa_module=librosa_module)
    bpm = float(metas.get("bpm") or DEFAULT_BPM)
    return bpm, metas["keyscale"], int(metas["timesignature"])


# Beat alignment: does the regenerated backing play the song's rhythm?
ALIGNMENT_SAMPLE_RATE = 22050
ALIGNMENT_HOP = 512
ALIGNMENT_WINDOW_SECONDS = 30.0
ALIGNMENT_MAX_LAG_SECONDS = 0.5


def _pearson(np, a, b) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def beat_alignment(reference_path: str, backing_path: str, librosa_module=None, numpy_module=None) -> dict:
    """How well the backing's rhythm follows the reference's.

    Returns {"onset_corr": float, "best_lag_ms": float, "windows": int}.
    onset_corr is the mean, over 30 s windows, of the Pearson correlation (lag
    0) between the two onset-strength envelopes. best_lag_ms is the mean, over
    windows, of the |lag| in ms that maximises that correlation within
    +/-0.5 s. A file shorter than one window counts as one window.
    """
    librosa = librosa_module
    if librosa is None:
        import librosa
    np = numpy_module
    if np is None:
        import numpy as np

    envelopes = []
    for path in (reference_path, backing_path):
        y, sr = librosa.load(path, sr=ALIGNMENT_SAMPLE_RATE, mono=True)
        envelopes.append(np.asarray(
            librosa.onset.onset_strength(y=y, sr=sr, hop_length=ALIGNMENT_HOP), dtype=float
        ))
    n = min(len(envelopes[0]), len(envelopes[1]))
    ref, back = envelopes[0][:n], envelopes[1][:n]

    frame_seconds = ALIGNMENT_HOP / ALIGNMENT_SAMPLE_RATE
    window = max(1, int(ALIGNMENT_WINDOW_SECONDS / frame_seconds))
    max_lag = int(ALIGNMENT_MAX_LAG_SECONDS / frame_seconds)
    starts = list(range(0, n - window + 1, window)) if n >= window else [0]

    corrs, lags = [], []
    for start in starts:
        a = ref[start:start + window]
        b = back[start:start + window]
        corrs.append(_pearson(np, a, b))
        best_lag, best_corr = 0, None
        for lag in range(-max_lag, max_lag + 1):
            # b[i + lag] against a[i], over the part where both exist.
            x = a[max(0, -lag):len(a) - max(0, lag)]
            z = b[max(0, lag):len(b) - max(0, -lag)]
            if len(x) < 2:
                continue
            c = _pearson(np, x, z)
            if best_corr is None or c > best_corr + 1e-12 or (abs(c - best_corr) <= 1e-12 and abs(lag) < abs(best_lag)):
                best_lag, best_corr = lag, c
        lags.append(abs(best_lag) * frame_seconds * 1000.0)

    return {
        "onset_corr": float(sum(corrs) / len(corrs)),
        "best_lag_ms": float(sum(lags) / len(lags)),
        "windows": len(starts),
    }
