"""Run a Whisper model on many clips (used by the baseline, test and demo)."""

from .audio_utils import SAMPLE_RATE


def load_whisper(model_name_or_path, device=None):
    """Load a Whisper model and its processor, in half precision on a GPU."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    dtype_arg = {"dtype": dtype} if _new_api() else {"torch_dtype": dtype}
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path, **dtype_arg
    ).to(device)
    model.eval()
    return model, processor


def _new_api():
    """transformers v5 renamed ``torch_dtype`` to ``dtype``."""
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


def transcribe(model, processor, arrays, batch_size=16, max_new_tokens=440):
    """Transcribe a list of 16 kHz audio arrays into Telugu text.

    ``language="te"`` and ``task="transcribe"`` tell Whisper to write
    Telugu, not to guess the language or translate into English.
    """
    import torch
    from tqdm.auto import tqdm

    texts = []
    for start in tqdm(range(0, len(arrays), batch_size), desc="transcribing"):
        batch = arrays[start:start + batch_size]
        features = processor.feature_extractor(
            batch, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(
                features, language="te", task="transcribe",
                max_new_tokens=max_new_tokens,
            )
        texts.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return texts
