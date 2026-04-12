#!/usr/bin/env python3
"""
Preprocess train_texts.txt and test_texts.txt for a dataset using
TopMost's Preprocess.parse(), filtering tokens to the existing vocabulary.

Follows the same approach as TopMost/tutorials/tutorial_preprocessing_datasets.ipynb
but adapted for datasets that use .txt files instead of .jsonlist.

Usage:
    cd /home/ducanhhh/Continual-baseline
    uv run python3 scripts/preprocess_texts.py --data_dir data/NIPS
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TopMost"))

from topmost import Preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/NIPS")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    vocab = (data_dir / "vocab.txt").read_text().splitlines()
    print(f"Vocab size: {len(vocab)}")

    preprocess = Preprocess(vocab_size=len(vocab), min_term=1, stopwords='snowball', verbose=True)

    for split in ("train", "test"):
        path = data_dir / f"{split}_texts.txt"
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        raw_texts = path.read_text(encoding='utf-8', errors='ignore').splitlines()
        print(f"\nProcessing {split}: {len(raw_texts)} docs")

        processed_texts, _ = preprocess.parse(raw_texts, vocab)

        with open(path, 'w', encoding='utf-8') as f:
            for t in processed_texts:
                f.write(t + '\n')

        nonempty = sum(1 for t in processed_texts if t.strip())
        print(f"  Saved. Non-empty: {nonempty}/{len(processed_texts)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
