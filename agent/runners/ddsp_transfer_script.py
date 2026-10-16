"""DDSP timbre transfer: render a monophonic solo as another instrument.

Standalone. Provisioning writes this file to /opt/ddsp/transfer.py and the
ddsp_timbre runner runs it with /opt/ddsp/venv/bin/python, a Python 3.10 venv
holding TensorFlow 2.11 and ddsp 3.7.0. It imports nothing from Pulsar: the
agent's torch Python and this venv share no packages.

Adapted from Magenta's timbre_transfer.ipynb (ddsp/colab/demos, Apache-2.0):
the gin length bindings, detect_notes, the octave shift toward the dataset's
mean pitch, the quantile loudness transform and shift_f0/shift_ld are the
notebook's own. The heavy imports live inside the functions that need them,
so the argument parsing and arithmetic can be tested without TensorFlow.
"""

import argparse
import os
import pickle
import sys
import time

MODEL_SAMPLE_RATE = 16000
CHECKPOINT = "ckpt-20000"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instrument", default="tenor_saxophone")
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--out", dest="output", required=True)
    parser.add_argument("--models-root", default="/opt/ddsp/models")
    # "auto": shift by whole octaves toward the model's training register
    # (needs --adjust 1). An integer: shift by that many octaves instead.
    parser.add_argument("--octave-shift", default="auto")
    parser.add_argument("--loudness-shift-db", type=float, default=0.0)
    parser.add_argument("--adjust", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-sample-rate", type=int, default=None)
    # The notebook's note-detection threshold and quiet-part attenuation (dB).
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--quiet", type=float, default=20.0)
    return parser.parse_args(argv)


def auto_octave_shift(target_mean_pitch, source_mean_pitch):
    """Whole octaves toward the dataset's mean pitch, rounded as the notebook does."""
    import math

    p_diff_octave = (target_mean_pitch - source_mean_pitch) / 12.0
    round_fn = math.floor if p_diff_octave > 1.5 else math.ceil
    return int(round_fn(p_diff_octave))


def clip_lengths(n_samples_train, time_steps_train, n_input_samples):
    """(time_steps, n_samples) for the whole clip, at the checkpoint's hop size."""
    hop_size = int(n_samples_train / time_steps_train)
    time_steps = int(n_input_samples / hop_size)
    return time_steps, time_steps * hop_size


def gin_length_bindings(time_steps, n_samples):
    return [
        "Harmonic.n_samples = {}".format(n_samples),
        "FilteredNoise.n_samples = {}".format(n_samples),
        "F0LoudnessPreprocessor.time_steps = {}".format(time_steps),
        "oscillator_bank.use_angular_cumsum = True",
    ]


def shift_ld(audio_features, ld_shift=0.0):
    audio_features["loudness_db"] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    import librosa
    import numpy as np

    audio_features["f0_hz"] *= 2.0 ** (pitch_shift)
    audio_features["f0_hz"] = np.clip(audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0))
    return audio_features


def load_mono_16k(path):
    """(audio [1, n] at 16 kHz, source rate, source channel count)."""
    import librosa
    import numpy as np
    import soundfile as sf

    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    channels = audio.shape[1]
    mono = audio.mean(axis=1)
    if source_rate != MODEL_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=MODEL_SAMPLE_RATE)
    return np.asarray(mono, dtype=np.float32)[np.newaxis, :], source_rate, channels


def transfer(args):
    import ddsp
    import ddsp.training
    import gin
    import numpy as np
    import soundfile as sf
    import librosa
    import tensorflow as tf  # noqa: F401 - loaded here so its startup log comes first
    from ddsp.training.postprocessing import detect_notes, fit_quantile_transform

    model_dir = os.path.join(args.models_root, args.instrument)
    gin_file = os.path.join(model_dir, "operative_config-0.gin")
    ckpt = os.path.join(model_dir, CHECKPOINT)
    for needed in (gin_file, ckpt + ".index", os.path.join(model_dir, "dataset_statistics.pkl")):
        if not os.path.exists(needed):
            raise SystemExit("model file missing: " + needed)

    with open(os.path.join(model_dir, "dataset_statistics.pkl"), "rb") as f:
        dataset_stats = pickle.load(f)

    audio, source_rate, channels = load_mono_16k(args.input)
    print("Loaded {}: {:.1f} s, {} Hz, {} channel(s)".format(
        args.input, audio.shape[1] / MODEL_SAMPLE_RATE, source_rate, channels), flush=True)

    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    print("Computing audio features (CREPE f0, loudness)...", flush=True)
    ddsp.spectral_ops.reset_crepe()
    start = time.time()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in audio_features.items()}
    audio_features["loudness_db"] = audio_features["loudness_db"].astype(np.float32)
    print("Audio features took %.1f seconds" % (time.time() - start), flush=True)

    time_steps, n_samples = clip_lengths(
        gin.query_parameter("Harmonic.n_samples"),
        gin.query_parameter("F0LoudnessPreprocessor.time_steps"),
        audio.shape[1],
    )
    with gin.unlock_config():
        gin.parse_config(gin_length_bindings(time_steps, n_samples))

    for key in ("f0_hz", "f0_confidence", "loudness_db"):
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]

    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
    manual_octaves = 0 if args.octave_shift == "auto" else int(args.octave_shift)

    if args.adjust:
        mask_on, note_on_value = detect_notes(
            audio_features["loudness_db"], audio_features["f0_confidence"], args.threshold)
        if np.any(mask_on):
            if args.octave_shift == "auto":
                pitch = ddsp.core.hz_to_midi(audio_features["f0_hz"])
                octaves = auto_octave_shift(dataset_stats["mean_pitch"], float(np.mean(pitch[mask_on])))
                print("Auto octave shift: {:+d}".format(octaves), flush=True)
                audio_features_mod = shift_f0(audio_features_mod, octaves)

            _, loudness_norm = fit_quantile_transform(
                audio_features["loudness_db"], mask_on, inv_quantile=dataset_stats["quantile_transform"])
            mask_off = np.logical_not(mask_on)
            loudness_norm[mask_off] -= args.quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
            loudness_norm = np.reshape(loudness_norm, audio_features["loudness_db"].shape)
            audio_features_mod["loudness_db"] = loudness_norm
        else:
            print("Skipping auto-adjust (no notes detected).", flush=True)
    else:
        print("Skipping auto-adjust (disabled).", flush=True)

    audio_features_mod = shift_ld(audio_features_mod, args.loudness_shift_db)
    audio_features_mod = shift_f0(audio_features_mod, manual_octaves)

    print("Restoring {} and running inference...".format(ckpt), flush=True)
    model = ddsp.training.models.Autoencoder()
    # verbose=False is expect_partial(): the checkpoint also holds the
    # optimizer's state, which inference never builds, and without it TF
    # prints several hundred "could not be found" warnings at exit - which
    # would fill job.log's tail and bury the lines that matter.
    model.restore(ckpt, verbose=False)
    # The notebook calls the model once to build it (the checkpoint restore is
    # deferred until the variables exist), then again for the real output.
    start = time.time()
    _ = model(audio_features_mod, training=False)
    print("Model build took %.1f seconds" % (time.time() - start), flush=True)
    start = time.time()
    outputs = model(audio_features_mod, training=False)
    audio_gen = np.asarray(model.get_audio_from_outputs(outputs), dtype=np.float32)
    print("Prediction took %.1f seconds" % (time.time() - start), flush=True)
    if audio_gen.ndim > 1:
        audio_gen = audio_gen[0]

    out_rate = args.output_sample_rate or source_rate
    if out_rate != MODEL_SAMPLE_RATE:
        audio_gen = librosa.resample(audio_gen, orig_sr=MODEL_SAMPLE_RATE, target_sr=out_rate)
    peak = float(np.max(np.abs(audio_gen))) if audio_gen.size else 0.0
    if peak > 1.0:
        # PCM cannot hold it; scale rather than clip the synth's transients.
        print("Peak {:.2f} above full scale; scaled to 0.99".format(peak), flush=True)
        audio_gen = audio_gen * (0.99 / peak)
    if channels > 1:
        audio_gen = np.repeat(audio_gen[:, np.newaxis], channels, axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, audio_gen, out_rate, subtype="PCM_24")
    print("Wrote {} ({} Hz, {} channel(s))".format(args.output, out_rate, channels), flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.octave_shift != "auto":
        try:
            int(args.octave_shift)
        except ValueError:
            raise SystemExit("--octave-shift must be 'auto' or an integer, got " + repr(args.octave_shift))
    transfer(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
