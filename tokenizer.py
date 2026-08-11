"""
A minimal character-level tokenizer.

Every unique character in the training text becomes one token. It's the
simplest possible tokenizer -- no external vocabulary needed, fully
transparent -- at the cost of the model needing more steps to learn since
each token carries very little information on its own (compare to a
subword tokenizer like GPT's, where "ing" or "the" might be a single token).
"""

import json


class CharTokenizer:
    def __init__(self, vocab=None):
        # vocab: a sorted list of unique characters seen in training data.
        # stoi = "string to integer", itos = "integer to string"
        self.vocab = vocab or []
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @classmethod
    def from_text(cls, text: str):
        vocab = sorted(list(set(text)))
        return cls(vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])