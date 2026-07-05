import os

from tinyllm.tokenizer import EOS_TOKEN, Tokenizer

FIXTURE_TEXT = (
    "One day, a little girl named Lily found a needle in her room.<|endoftext|>"
    "She wanted to play with her friend in the park.<|endoftext|>"
    "The cat sat on the mat and looked at the sun.<|endoftext|>"
)


def train_fixture_tokenizer(tmp_path, vocab_size=300):
    input_path = tmp_path / "fixture.txt"
    input_path.write_text(FIXTURE_TEXT, encoding="utf-8")
    return Tokenizer.train(str(input_path), vocab_size=vocab_size)


def test_roundtrip(tmp_path):
    tok = train_fixture_tokenizer(tmp_path)
    sentence = "The cat sat on the mat."
    ids = tok.encode(sentence)
    assert tok.decode(ids) == sentence


def test_eos_token_single_id(tmp_path):
    tok = train_fixture_tokenizer(tmp_path)
    ids = tok.encode(EOS_TOKEN)
    assert ids == [tok.eos_id]


def test_vocab_size_matches_arg(tmp_path):
    vocab_size = 300
    tok = train_fixture_tokenizer(tmp_path, vocab_size=vocab_size)
    assert tok.vocab_size == vocab_size


def test_save_and_load_roundtrip(tmp_path):
    tok = train_fixture_tokenizer(tmp_path)
    save_path = tmp_path / "tokenizer.json"
    tok.save(str(save_path))
    assert os.path.exists(save_path)

    loaded = Tokenizer.load(str(save_path))
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("The cat sat.") == tok.encode("The cat sat.")
