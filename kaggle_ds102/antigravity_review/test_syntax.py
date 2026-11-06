"""Syntax Verification Suite for antigravity_review codebase.

Runs locally on Mac to verify AST parsing, compilation, and safety guards.
NO TRAINING IS PERFORMED.
"""

import ast
import compileall
import importlib.util
import os
import sys
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent


def verify_syntax():
    py_files = sorted(REVIEW_DIR.glob('*.py'))
    print(f"Verifying syntax across {len(py_files)} Python files in {REVIEW_DIR}...")
    errors = []

    for f in py_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                source = fp.read()
            # 1. AST Parse
            ast.parse(source, filename=str(f))
            # 2. Compile Check
            compile(source, str(f), 'exec')
            print(f"  [PASS] {f.name} (AST + Bytecode compiled)")
        except Exception as e:
            errors.append((f.name, str(e)))
            print(f"  [FAIL] {f.name}: {e}")

    if errors:
        print(f"\nFound {len(errors)} syntax errors!")
        sys.exit(1)
    else:
        print("\nAll files passed AST parsing and compilation successfully!")


def verify_guards():
    """Verify that execution guards correctly prevent running on local Mac."""
    print("\nVerifying Colab execution guard...")
    from colab_runner import guard as colab_guard
    from h1_intent_features import guard as h1_guard
    from h2_supervised_subwords import guard as h2_guard

    for name, g in [('colab_runner', colab_guard), ('h1_intent_features', h1_guard), ('h2_supervised_subwords', h2_guard)]:
        try:
            g()
            print(f"  [FAIL] {name} guard did not raise on Mac!")
            sys.exit(1)
        except RuntimeError as e:
            print(f"  [PASS] {name} guard correctly blocked local execution: '{e}'")


if __name__ == '__main__':
    verify_syntax()
    verify_guards()
