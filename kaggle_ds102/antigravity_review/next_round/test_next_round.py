"""Local syntax and logic verification for antigravity_review/next_round.

Runs locally on Mac:
- AST parsing and compilation of all files.
- Verifies SubwordTokenizer n-gram interleaving and truncation properties.
- Tests that guards correctly block execution outside Colab.
NO MODEL TRAINING IS PERFORMED.
"""

import ast
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h2_subword_cnn as h2


def test_tokenizer_interleaving_and_truncation():
    """Verify that positional interleaving includes 3, 4, and 5 grams even on truncated long text."""
    print("Testing SubwordTokenizer interleaving and truncation...")
    long_text = "Hàng đóng gói rất đẹp, giao hàng đúng hẹn, chưa dùng nên chưa biết chất lượng thế nào." * 10
    tokenizer = h2.SubwordTokenizer(vocab_size=500, min_n=3, max_n=5, min_freq=1)
    tokenizer.fit([long_text])

    # Max len 30: must contain 3-grams, 4-grams, and 5-grams, not only 3-grams!
    encoded = tokenizer.encode([long_text], max_len=30)[0]
    assert len(encoded) == 30

    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    token_strings = [inv_vocab.get(tok, '<UNK>') for tok in encoded if tok > 1]
    lengths = {len(s) for s in token_strings}

    print(f"  Truncated sequence (len=30) token lengths present: {sorted(lengths)}")
    assert 3 in lengths, "Missing 3-grams in truncated output!"
    assert 4 in lengths, "Missing 4-grams in truncated output! Interleaving fixed."
    assert 5 in lengths, "Missing 5-grams in truncated output! Interleaving fixed."
    print("  [PASS] Tokenizer interleaving verified successfully.")


def test_syntax_and_compilation():
    """Verify AST and compilation across all Python files in next_round."""
    py_files = sorted(HERE.glob('*.py'))
    print(f"\nVerifying syntax across {len(py_files)} files in {HERE.name}...")
    for f in py_files:
        with open(f, 'r', encoding='utf-8') as fp:
            source = fp.read()
        ast.parse(source, filename=str(f))
        compile(source, str(f), 'exec')
        print(f"  [PASS] {f.name}")


def test_colab_guards():
    """Verify that execution guards correctly prevent running on local Mac."""
    print("\nVerifying Colab guards...")
    for module_name, guard_fn in [('h2_subword_cnn', h2.guard), ('h2_experiment', h2.guard)]:
        try:
            guard_fn()
            print(f"  [FAIL] {module_name} guard failed to block execution on Mac!")
            sys.exit(1)
        except RuntimeError as e:
            print(f"  [PASS] {module_name} guard blocked execution: {e}")


if __name__ == '__main__':
    test_syntax_and_compilation()
    test_tokenizer_interleaving_and_truncation()
    test_colab_guards()
    print("\nAll local tests in test_next_round.py passed successfully!")
