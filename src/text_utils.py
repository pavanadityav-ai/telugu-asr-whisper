"""Text normalisation for Telugu speech recognition.

Why not use Whisper's own text normaliser?
Whisper's ``BasicTextNormalizer`` deletes Unicode "mark" characters.
In Telugu, vowel signs such as ా, ి, ు are marks, so that normaliser
turns "నాయకత్వం" into broken text and makes the error rate meaningless.
This normaliser keeps every Telugu letter and vowel sign and removes only
punctuation, symbols and invisible formatting characters.
"""

import re
import unicodedata

TELUGU_START, TELUGU_END = 0x0C00, 0x0C7F


def normalize_text(text: str) -> str:
    """Clean a sentence so that reference and prediction are compared fairly.

    Steps: Unicode NFC form, lower-case (only affects English letters),
    replace punctuation and symbols with spaces, drop invisible characters
    such as the zero-width non-joiner, and collapse repeated spaces.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S")):  # punctuation, symbols, danda (।)
            out.append(" ")
        elif cat == "Cf":  # invisible formatting characters
            continue
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def is_telugu_char(ch: str) -> bool:
    return TELUGU_START <= ord(ch) <= TELUGU_END


def script_profile(text: str) -> dict:
    """Share of letters that are Telugu, Latin (English) or something else.

    Useful to spot when the model answers in the wrong script, for example
    writing English or Hindi letters instead of Telugu.
    """
    letters = [ch for ch in text if unicodedata.category(ch)[0] in ("L", "M")]
    if not letters:
        return {"telugu": 0.0, "latin": 0.0, "other": 0.0}
    n = len(letters)
    telugu = sum(is_telugu_char(ch) for ch in letters)
    latin = sum(ch.isascii() for ch in letters)
    return {
        "telugu": telugu / n,
        "latin": latin / n,
        "other": (n - telugu - latin) / n,
    }
