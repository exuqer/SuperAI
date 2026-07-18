"""Hierarchical tokenization for recursive nebula system."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


# Regex for tokenizing Russian and Latin words (including hyphens and apostrophes)
TOKEN_OR_PUNCT_RE = re.compile(
    r"[0-9A-Za-zА-Яа-яЁё]+(?:['-][0-9A-Za-zА-Яа-яЁё]+)*|[.,!?;:]",
    re.IGNORECASE,
)

# Punctuation tokens to filter out
PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}

# Sentence boundaries must be found in the source text, before whitespace is
# normalized.  A newline is a boundary in its own right; terminal punctuation
# closes a sentence even when the next sentence starts immediately after it.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\s+|(?=[^\s.!?]))|\n+")


@dataclass
class CharacterToken:
    """Single character with position info."""
    value: str
    position: int
    normalized: str = ""
    
    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.value.casefold()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "normalized": self.normalized,
            "position": self.position,
        }


@dataclass
class WordToken:
    """Word form with character breakdown."""
    text: str
    normalized: str
    position: int
    characters: List[CharacterToken] = field(default_factory=list)
    sentence_index: int = 0
    token_index_in_sentence: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "position": self.position,
            "sentence_index": self.sentence_index,
            "token_index_in_sentence": self.token_index_in_sentence,
            "characters": [c.to_dict() for c in self.characters],
        }


@dataclass
class SentenceTokens:
    """Sentence with tokenized words."""
    text: str
    index: int
    tokens: List[WordToken] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "tokens": [t.to_dict() for t in self.tokens],
        }


@dataclass
class TokenizationResult:
    """Full hierarchical tokenization result."""
    text: str
    sentences: List[SentenceTokens] = field(default_factory=list)
    all_tokens: List[WordToken] = field(default_factory=list)
    all_characters: List[CharacterToken] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sentences": [s.to_dict() for s in self.sentences],
            "tokens": [t.to_dict() for t in self.all_tokens],
            "total_tokens": len(self.all_tokens),
            "total_characters": len(self.all_characters),
        }


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and stripping."""
    return " ".join(str(text or "").replace("\r", " ").split()).strip()


def canonical_token(token: str) -> str:
    """Convert token to canonical form (lowercase)."""
    return normalize_text(str(token or "")).casefold()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    source = str(text or "")
    if not source.strip():
        return []
    chunks = [
        chunk.strip()
        for chunk in SENTENCE_SPLIT_RE.split(source)
        if chunk.strip()
    ]
    return chunks or [normalize_text(source)]


def tokenize(text: str) -> List[str]:
    """Simple tokenization returning only word tokens (backward compatibility)."""
    return [
        item["token"]
        for item in tokenize_with_surfaces(text)
        if item["token"] not in PUNCT_TOKENS
    ]


def tokenize_with_surfaces(text: str) -> List[Dict[str, str]]:
    """Tokenize keeping surface forms (backward compatibility)."""
    items: List[Dict[str, str]] = []
    for match in TOKEN_OR_PUNCT_RE.finditer(str(text or "")):
        token = canonical_token(match.group(0))
        if token:
            items.append({"surface": token, "token": token})
    return items


def tokenize_hierarchical(text: str) -> TokenizationResult:
    """
    Full hierarchical tokenization:
    text -> sentences -> words -> characters
    Preserves order at all levels for condensation.
    """
    result = TokenizationResult(text=normalize_text(text))
    sentences = split_sentences(text)
    
    global_token_pos = 0
    global_char_pos = 0
    
    for sent_idx, sent_text in enumerate(sentences):
        sentence = SentenceTokens(text=sent_text, index=sent_idx)
        
        # Find word tokens in sentence
        word_matches = list(TOKEN_OR_PUNCT_RE.finditer(sent_text))
        token_idx_in_sent = 0
        
        for match in word_matches:
            token_text = match.group(0)
            token_normalized = canonical_token(token_text)
            
            if token_normalized in PUNCT_TOKENS:
                continue
            
            # Create character tokens for this word
            characters = []
            for char_idx, char in enumerate(token_normalized):
                char_token = CharacterToken(
                    value=char,
                    position=global_char_pos,
                    normalized=char
                )
                characters.append(char_token)
                result.all_characters.append(char_token)
                global_char_pos += 1
            
            word_token = WordToken(
                text=token_text,
                normalized=token_normalized,
                position=global_token_pos,
                characters=characters,
                sentence_index=sent_idx,
                token_index_in_sentence=token_idx_in_sent,
            )
            
            sentence.tokens.append(word_token)
            result.all_tokens.append(word_token)
            global_token_pos += 1
            token_idx_in_sent += 1
        
        if sentence.tokens:
            result.sentences.append(sentence)
    
    return result


def get_character_sequence(word_token: WordToken) -> List[str]:
    """Get ordered character sequence for a word (for condensation signature)."""
    return [c.normalized for c in word_token.characters]


def get_word_signature(word_token: WordToken, layer_name: str = "word_form") -> str:
    """Generate signature for condensation candidate."""
    char_ids = "|".join(str(id(c)) for c in word_token.characters)  # placeholder, will use cloud IDs
    return f"{layer_name}:{word_token.normalized}:{char_ids}"
