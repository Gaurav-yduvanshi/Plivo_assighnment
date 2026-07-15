"""Corpus-trained byte-fallback tokenizer.

This version keeps the exact UTF-8 round-trip contract but trains quickly by
scoring frequent byte substrings from whitespace-attached text units.
"""
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_VOCAB_SIZE = 8192
DEFAULT_MAX_UNITS = 50000
DEFAULT_MAX_NGRAM = 16
DEFAULT_MAX_TOKEN_BYTES = 64
ARTIFACT_NAME = "tokenizer.json"
CHUNK_RE = re.compile(r"\s+|\S+")


def _artifact_path():
    return Path(__file__).with_name(ARTIFACT_NAME)


def _attach_leading_spaces(text):
    units = []
    pending_space = b""
    for chunk in CHUNK_RE.findall(text):
        chunk_bytes = chunk.encode("utf-8")
        if chunk.isspace():
            pending_space += chunk_bytes
        else:
            units.append(pending_space + chunk_bytes)
            pending_space = b""
    if pending_space:
        units.append(pending_space)
    return units


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="strict")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"type": "byte", "vocab_size": self.vocab_size}, handle)


class SubstringTokenizer:
    def __init__(self, token_bytes):
        self.token_bytes = [bytes(token) for token in token_bytes]
        self.vocab_size = 256 + len(self.token_bytes)
        self._trie = {}
        for index, token in enumerate(self.token_bytes, start=256):
            node = self._trie
            for byte in token:
                node = node.setdefault(byte, {})
            node["id"] = index

    def encode(self, text):
        data = text.encode("utf-8")
        ids = []
        i = 0
        n = len(data)
        while i < n:
            node = self._trie
            best_id = None
            best_end = None
            j = i
            while j < n:
                next_node = node.get(data[j])
                if next_node is None:
                    break
                node = next_node
                j += 1
                token_id = node.get("id")
                if token_id is not None:
                    best_id = token_id
                    best_end = j
            if best_id is None:
                ids.append(data[i])
                i += 1
            else:
                ids.append(best_id)
                i = best_end
        return ids

    def decode(self, ids):
        out = bytearray()
        for token_id in ids:
            if token_id < 256:
                out.append(token_id)
            else:
                out.extend(self.token_bytes[token_id - 256])
        return out.decode("utf-8", errors="strict")

    def save(self, path):
        payload = {
            "type": "substring",
            "vocab_size": self.vocab_size,
            "token_hexes": [token.hex() for token in self.token_bytes],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("type") != "substring":
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')!r}")
        token_bytes = [bytes.fromhex(item) for item in payload["token_hexes"]]
        return cls(token_bytes)

    @classmethod
    def train(cls, text, vocab_size=DEFAULT_VOCAB_SIZE,
              max_units=DEFAULT_MAX_UNITS,
              max_ngram=DEFAULT_MAX_NGRAM,
              max_token_bytes=DEFAULT_MAX_TOKEN_BYTES):
        vocab_size = max(256, int(vocab_size))
        target_tokens = vocab_size - 256
        unit_counts = Counter(_attach_leading_spaces(text))

        candidate_scores = Counter()
        for unit_bytes, freq in unit_counts.most_common(max_units):
            if freq < 2 or len(unit_bytes) < 2:
                continue
            if len(unit_bytes) <= max_token_bytes:
                candidate_scores[unit_bytes] += freq * (len(unit_bytes) - 1) * len(unit_bytes)

            span = min(len(unit_bytes), max_ngram)
            for start in range(len(unit_bytes)):
                end_limit = min(len(unit_bytes), start + span)
                for end in range(start + 2, end_limit + 1):
                    token = unit_bytes[start:end]
                    if len(token) > max_token_bytes:
                        continue
                    bonus = 1.5 if token[:1].isspace() else 1.0
                    candidate_scores[token] += int(freq * (len(token) - 1) * bonus)

        learned_tokens = []
        seen = set()
        for token_bytes, score in sorted(
                candidate_scores.items(),
                key=lambda item: (item[1], len(item[0]), item[0]),
                reverse=True):
            if len(learned_tokens) >= target_tokens:
                break
            if score < 2 or len(token_bytes) < 2:
                continue
            if token_bytes in seen:
                continue
            seen.add(token_bytes)
            learned_tokens.append(token_bytes)

        return cls(learned_tokens)


def load(path=None, training_text=None, vocab_size=DEFAULT_VOCAB_SIZE,
         max_units=DEFAULT_MAX_UNITS,
         max_ngram=DEFAULT_MAX_NGRAM,
         max_token_bytes=DEFAULT_MAX_TOKEN_BYTES):
    """Load the tokenizer used by evaluate.py."""
    artifact_path = Path(path) if path is not None else _artifact_path()
    if training_text is not None:
        tokenizer = SubstringTokenizer.train(
            training_text,
            vocab_size=vocab_size,
            max_units=max_units,
            max_ngram=max_ngram,
            max_token_bytes=max_token_bytes,
        )
        tokenizer.save(artifact_path)
        return tokenizer
    if artifact_path.exists():
        return SubstringTokenizer.load(artifact_path)
    return ByteTokenizer()