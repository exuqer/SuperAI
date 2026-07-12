"""Tokenization module for Russian/Latin words - ported from alg2 branch"""
import re
from typing import List

# Regex for tokenizing Russian and Latin words (including hyphens and apostrophes)
# Matches words with letters from Cyrillic (Russian) and Latin alphabets
TOKEN_OR_PUNCT_RE = re.compile(
    r"[0-9A-Za-zА-Яа-яЁё]+(?:['-][0-9A-Za-zА-Яа-яЁё]+)*|[.,!?;:]",
    re.IGNORECASE,
)

# Punctuation tokens to filter out
PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}

# Sentence splitting regex
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and stripping."""
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def canonical_token(token: str) -> str:
    """Convert token to canonical form (lowercase)."""
    cleaned = normalize_text(str(token or "")).casefold()
    return cleaned


def tokenize_with_surfaces(text: str) -> List[dict]:
    """Tokenize text keeping surface forms."""
    items: List[dict] = []
    for match in TOKEN_OR_PUNCT_RE.finditer(str(text or "")):
        token = canonical_token(match.group(0))
        if token:
            items.append({"surface": token, "token": token})
    return items


def tokenize(text: str) -> List[str]:
    """Tokenize text, filtering out punctuation and returning only words."""
    return [
        item["token"]
        for item in tokenize_with_surfaces(text)
        if item["token"] not in PUNCT_TOKENS
    ]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    normalized = str(text or "").replace("\r", "\n").strip()
    if not normalized:
        return []
    chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(normalized) if chunk.strip()]
    return chunks or [normalized]