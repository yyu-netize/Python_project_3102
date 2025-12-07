# pii_redaction.py
import re
import json

# ---------------------------
# Define regex patterns
# ---------------------------

PII_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"\b(\+?\d[\d\-\s]{7,}\d)\b"),
    "ID_NUMBER": re.compile(r"\b([A-Z0-9]{6,12})\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# ---------------------------
# Redaction function
# ---------------------------

def redact_text(text: str) -> str:
    """
    Replace PII-like patterns with <PII_TYPE_REDACTED>.
    """
    if not text:
        return text
    
    for label, pattern in PII_PATTERNS.items():
        placeholder = f"<{label}_REDACTED>"
        text = pattern.sub(placeholder, text)
    return text


def redact_corpus(chunks):
    """
    chunks: list of dicts, each containing:
        - id
        - text
    Return a new list with redacted text.
    """
    new_chunks = []
    for c in chunks:
        new_chunks.append({
            "id": c["id"],
            "text": redact_text(c["text"])
        })
    return new_chunks
