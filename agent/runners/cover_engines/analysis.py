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
