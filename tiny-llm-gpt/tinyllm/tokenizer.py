"""Thin wrapper around the `tokenizers` library so the rest of the codebase
never imports `tokenizers` directly.
"""
from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"


class Tokenizer:
    def __init__(self, hf_tokenizer: _HFTokenizer):
        self._tok = hf_tokenizer

    @classmethod
    def train(cls, input_path: str, vocab_size: int) -> "Tokenizer":
        hf_tok = _HFTokenizer(models.BPE())
        hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        hf_tok.train([input_path], trainer)
        return cls(hf_tok)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        return cls(_HFTokenizer.from_file(path))

    def save(self, path: str) -> None:
        self._tok.save(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(EOS_TOKEN)
