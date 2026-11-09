"""Loading the FLEURS Telugu data and turning audio bytes into numbers."""

import io

import numpy as np

SAMPLE_RATE = 16_000  # Whisper expects 16,000 samples per second
MAX_SECONDS = 30  # Whisper reads audio in 30-second windows


def load_fleurs_telugu(split=None):
    """Load Google FLEURS Telugu ("te_in") with audio left as raw bytes.

    Keeping the audio undecoded (``decode=False``) makes loading fast and
    works the same way across ``datasets`` library versions. We decode a
    clip only when we need it, with ``decode_audio``.
    """
    from datasets import Audio, load_dataset

    try:
        ds = load_dataset("google/fleurs", "te_in", split=split)
    except Exception:  # older dataset-script versions of the repo
        ds = load_dataset("google/fleurs", "te_in", split=split,
                          revision="refs/convert/parquet")
    return ds.cast_column("audio", Audio(decode=False))


def decode_audio(audio) -> np.ndarray:
    """Turn one stored clip ({"bytes": ..., "path": ...}) into a float array.

    Returns mono audio at 16 kHz with values between -1 and 1.
    """
    import soundfile as sf

    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    data, sr = sf.read(source, dtype="float32")
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data


def seconds(num_samples: int) -> float:
    return num_samples / SAMPLE_RATE
