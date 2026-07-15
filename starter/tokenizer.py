"""Tokenizer with byte fallback.

The trained version stores frequent UTF-8 byte substrings from the provided
corpus and uses greedy longest-match encoding. Any unseen text still falls
back to raw bytes, so decode(encode(text)) stays exactly equal to text.
"""
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_VOCAB_SIZE = 4096
DEFAULT_MAX_CANDIDATES = 50000
DEFAULT_MAX_NGRAM = 12
DEFAULT_MAX_CHUNK_BYTES = 64
ARTIFACT_NAME = "tokenizer.json"
CHUNK_RE = re.compile(r"\s+|\S+")


def _artifact_path():
    return Path(__file__).with_name(ARTIFACT_NAME)


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
              max_candidates=DEFAULT_MAX_CANDIDATES,
              max_ngram=DEFAULT_MAX_NGRAM,
              max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES):
        vocab_size = max(256, int(vocab_size))
        target_learned = vocab_size - 256
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

        chunk_counts = Counter(units)

        candidate_scores = Counter()
        for chunk_bytes, freq in chunk_counts.most_common(max_candidates):
            if freq < 2:
                continue
            if len(chunk_bytes) < 2:
                continue
            if len(chunk_bytes) > max_chunk_bytes:
                continue
            whole_score = freq * (len(chunk_bytes) - 1) * len(chunk_bytes)
            candidate_scores[chunk_bytes] += whole_score
            span = min(len(chunk_bytes), max_ngram)
            for start in range(len(chunk_bytes)):
                end_limit = min(len(chunk_bytes), start + span)
                for end in range(start + 2, end_limit + 1):
                    token = chunk_bytes[start:end]
                    candidate_scores[token] += freq * (len(token) - 1)

        learned_tokens = []
        seen = set()
        for token_bytes, score in sorted(
                candidate_scores.items(),
                key=lambda item: (item[1], len(item[0]), item[0]),
                reverse=True):
            if len(learned_tokens) >= target_learned:
                break
            if len(token_bytes) < 2 or score < 2:
                continue
            if token_bytes in seen:
                continue
            seen.add(token_bytes)
            learned_tokens.append(token_bytes)

        return cls(learned_tokens)


def load(path=None, training_text=None, vocab_size=DEFAULT_VOCAB_SIZE,
         max_candidates=DEFAULT_MAX_CANDIDATES,
         max_ngram=DEFAULT_MAX_NGRAM,
         max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES):
    """Load the tokenizer used by evaluate.py.

    If training_text is provided, a fresh tokenizer is trained from that text
    and written to the default on-disk artifact.
    """
    artifact_path = Path(path) if path is not None else _artifact_path()
    if training_text is not None:
        tokenizer = SubstringTokenizer.train(
            training_text,
            vocab_size=vocab_size,
            max_candidates=max_candidates,
            max_ngram=max_ngram,
            max_chunk_bytes=max_chunk_bytes,
        )
        tokenizer.save(artifact_path)
        return tokenizer
    if artifact_path.exists():
        return SubstringTokenizer.load(artifact_path)
    return ByteTokenizer()