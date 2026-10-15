import glob
import os
import threading

from agent.runners.base import Runner

try:  # pragma: no cover - only importable where basic-pitch is installed
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save
except ImportError:  # tests patch these names directly
    ICASSP_2022_MODEL_PATH = None
    predict_and_save = None


# Basic Pitch's own thresholds. Forwarded only when a job sets them, so
# anything unspecified keeps the library default rather than a guess here.
OPTIONAL_PARAMS = (
    "onset_threshold",
    "frame_threshold",
    "minimum_note_length",
    "minimum_frequency",
    "maximum_frequency",
    "multiple_pitch_bends",
    "melodia_trick",
    "midi_tempo",
)


class AudioToMidiRunner(Runner):
    """Transcribes audio to MIDI with Spotify's Basic Pitch.

    Polyphonic, pitch-bend aware, instrument-agnostic and permissively
    licensed, which is why it's preferred over the monophonic alternatives.

    Exists mainly to serve melody preservation: if a cover model drifts off
    the tune, the recovery path is to isolate the vocal with Demucs,
    transcribe the melody here, and either condition on the MIDI or use
    `sonify` to render a clean synthesised version of that melody as a
    reference for the cover step.
    """

    def __init__(self, workdir: str = "/workspace"):
        self._workdir = workdir
        self._status = "idle"
        self._output_path: str | None = None
        self._log_path = os.path.join(workdir, "job.log")

    def submit(self, input_path: str, params: dict) -> None:
        threading.Thread(target=self._execute, args=(input_path, params), daemon=True).start()

    def _execute(self, input_path: str, params: dict) -> None:
        self._status = "running"
        try:
            output_dir = os.path.join(self._workdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Rendering MIDI back to audio roughly doubles the work, so it's
            # opt-in: only wanted when the melody is being handed to another
            # model rather than inspected as notes.
            sonify = bool(params.get("sonify", False))

            optional = {k: params[k] for k in OPTIONAL_PARAMS if k in params}
            predict_and_save(
                [input_path],
                output_dir,
                True,   # save_midi
                sonify,
                False,  # save_model_outputs - large npz, nothing consumes it
                False,  # save_notes - the .mid already carries the notes
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                **optional,
            )

            # Whichever artifact the caller can actually chain onwards: a
            # .mid is useless as audio input to a downstream model, so when
            # sonifying, the rendered wav is the real output.
            pattern = "*.wav" if sonify else "*.mid"
            matches = sorted(glob.glob(os.path.join(output_dir, pattern)))
            if matches:
                self._output_path = matches[0]
                self._status = "done"
            else:
                self._log(f"basic-pitch produced no {pattern} in {output_dir}")
                self._status = "failed"
                self._output_path = None
        except Exception as exc:
            self._status = "failed"
            self._output_path = None
            self._log(f"_execute failed: {exc}")

    def _log(self, message: str) -> None:
        try:
            with open(self._log_path, "a") as log_file:
                log_file.write(f"\n[agent] {message}\n")
        except OSError:
            pass

    def status(self) -> str:
        return self._status

    def output_path(self) -> str | None:
        return self._output_path

    def log_tail(self, lines: int = 100) -> str:
        try:
            with open(self._log_path) as log_file:
                return "".join(log_file.readlines()[-lines:])
        except FileNotFoundError:
            return ""
