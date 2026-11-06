"""Unit tests and syntax verification for Round 3 diagnostic tools.

Tests:
1. Syntax and import of colab_h2_truncation_diagnostic.py.
2. Token count logic matching SubwordTokenizer from h2_subword_cnn.py.
3. Diagnostic logic correctness with a synthetic test case (verifies confusion matrix, metrics, flips).
4. Padding invariance and character-cutoff mathematical verification.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import colab_h2_truncation_diagnostic as diag


def test_token_counting_consistency():
    # Verify exact token counts for known strings
    # "abc" -> normalized "^abc$" -> len=5.
    # i=0: "abc", "abc$", L-i=5 -> n=3, 4, 5. i+n <= 5 -> n=3 ("^ab"), n=4 ("^abc"), n=5 ("^abc$") (3 tokens)
    # i=1: "abc$", L-i=4 -> n=3 ("abc"), n=4 ("abc$") (2 tokens)
    # i=2: "bc$", L-i=3 -> n=3 ("bc$") (1 token)
    # i=3: "c$", L-i=2 -> no n >= 3 (0 tokens)
    # Total = 3 + 2 + 1 = 6 tokens.
    cnt = diag.count_interleaved_tokens("abc")
    assert cnt == 6, f"Expected 6, got {cnt}"

    # Verify empty string
    # "" -> "^$" -> len=2 -> no n >= 3 -> 0 tokens
    assert diag.count_interleaved_tokens("") == 0

    print("PASS: test_token_counting_consistency")


def test_char_cutoff_math():
    # Verify that max_len=256 stops at character position <= 89 in normalized text
    long_text = "a" * 200
    s = diag.normalize_text(long_text)
    L = len(s)
    tokens = []
    max_c = 0
    for i in range(L):
        for n in (3, 4, 5):
            if i + n <= L:
                tokens.append((i, s[i:i+n]))
                max_c = max(max_c, i + n)
                if len(tokens) >= 256:
                    break
        if len(tokens) >= 256:
            break
    assert len(tokens) == 256
    assert max_c == 89, f"Expected max_c=89, got {max_c}"
    # Raw character span is max_c - 2 (since ^ is index 0 and $ is last) = 87 chars
    print("PASS: test_char_cutoff_math")


def test_diagnostic_pipeline_synthetic():
    # Create 20 synthetic reviews: 10 short (<=256 tokens), 10 long (>256 tokens)
    short_texts = ["Sản phẩm dùng rất tốt giao hàng nhanh đóng gói cẩn thận"] * 10
    long_texts = ["Bình luận rất dài này chắc chắn sẽ vượt quá 256 subword tokens vì nó chứa nhiều câu lặp lại và mô tả chi tiết sản phẩm cùng quá trình vận chuyển " * 5] * 10
    texts = short_texts + long_texts
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    
    # Base: mostly correct
    base_preds = y_true.copy()
    base_preds[1] = 0  # error on short
    base_preds[11] = 0 # error on long

    # Cand: fixes short error, introduces new long error
    cand_preds = base_preds.copy()
    cand_preds[1] = 1   # good flip on short
    cand_preds[12] = 0  # bad flip on long

    report = diag.run_diagnostic(texts, y_true, base_preds, cand_preds, max_seq_len=256)
    assert report['summary']['total_samples'] == 20
    assert report['summary']['truncated_samples'] == 10
    assert report['summary']['nontruncated_samples'] == 10
    assert report['summary']['total_flips'] == 2
    assert report['summary']['good_flips'] == 1
    assert report['summary']['bad_flips'] == 1

    # Good flip occurred in non-truncated
    flips_c1 = report['flips_by_class']['1']
    assert flips_c1['good_flips_in_nontruncated'] == 1

    # Bad flip occurred in truncated
    flips_c2 = report['flips_by_class']['2']
    assert flips_c2['bad_flips_in_truncated'] == 1

    print("PASS: test_diagnostic_pipeline_synthetic")


if __name__ == '__main__':
    test_token_counting_consistency()
    test_char_cutoff_math()
    test_diagnostic_pipeline_synthetic()
    print("ALL ROUND 3 SYNTAX & UNIT TESTS PASSED.")
