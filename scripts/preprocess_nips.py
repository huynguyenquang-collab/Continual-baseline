"""
Preprocess NIPS corpus for CoNTM baseline.

Steps:
1. Read papers.csv (id, year, title, abstract, full_text)
2. Split full_text into paragraphs (each paragraph = 1 document)
3. Group years into 3-year bins → 11 timestamps (1987-2019)
4. Tokenize with spaCy (lowercase, remove stopwords/punct/numbers)
5. Build BoW with sklearn CountVectorizer (min_df=0.05%, max_df=95%)
6. Train/test split (90/10, stratified by timestamp)
7. Save: train_bow.npz, test_bow.npz, vocab.txt, times, texts, time2id

Target statistics (from paper Table 2):
  - ~276,657 documents
  - ~6,278 vocab
  - 11 timestamps

Usage:
    python scripts/preprocess_nips.py
    python scripts/preprocess_nips.py --input data/NIPS_raw/papers.csv --output-dir data/NIPS
"""

import csv
import sys
import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# ── Year → timestamp bin mapping (3-year windows) ────────────────────────────
YEAR_BINS = [
    (1987, 1989),  # T0
    (1990, 1992),  # T1
    (1993, 1995),  # T2
    (1996, 1998),  # T3
    (1999, 2001),  # T4
    (2002, 2004),  # T5
    (2005, 2007),  # T6
    (2008, 2010),  # T7
    (2011, 2013),  # T8
    (2014, 2016),  # T9
    (2017, 2019),  # T10
]


def year_to_timestamp(year: int) -> int:
    """Map a publication year to a timestamp index."""
    for i, (lo, hi) in enumerate(YEAR_BINS):
        if lo <= year <= hi:
            return i
    return -1


def split_into_paragraphs(text: str, min_words: int = 15) -> list[str]:
    """Split full text into paragraphs with at least min_words words."""
    raw_paras = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw_paras:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        if len(p.split()) >= min_words:
            paras.append(p)
    return paras


def tokenize_spacy(texts: list[str], batch_size: int = 10000) -> list[list[str]]:
    """Tokenize with spaCy blank English tokenizer.

    Lowercase, remove stopwords, punctuation, numbers.
    Keep only alphabetic tokens of length >= 3.
    """
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS

    nlp = spacy.blank("en")
    nlp.max_length = 2_000_000
    n_cpus = min(os.cpu_count() or 1, 16)
    print(f"  Using spaCy blank tokenizer with {n_cpus} processes")

    all_tokens = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_cpus),
        total=len(texts),
        desc="Tokenizing",
    ):
        tokens = []
        for tok in doc:
            t = tok.text.lower().strip()
            if t in STOP_WORDS or tok.is_punct or tok.is_space:
                continue
            if len(t) >= 3 and t.isalpha():
                tokens.append(t)
        all_tokens.append(tokens)
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIPS corpus for CoNTM")
    parser.add_argument("--input", default="data/NIPS_raw/papers.csv",
                        help="Path to papers.csv")
    parser.add_argument("--output-dir", default="data/NIPS",
                        help="Output directory")
    parser.add_argument("--min-para-words", type=int, default=40,
                        help="Min words per paragraph (paper gets ~276K docs with 40)")
    parser.add_argument("--min-df", type=float, default=0.0005,
                        help="min_df for CountVectorizer (fraction)")
    parser.add_argument("--max-df", type=float, default=0.95,
                        help="max_df for CountVectorizer (fraction)")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation set fraction (paper uses 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Read papers, split into paragraphs ───────────────────────────
    print("Step 1: Reading papers and splitting into paragraphs...")
    documents = []
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading papers"):
            year = int(row["year"])
            ts = year_to_timestamp(year)
            if ts < 0:
                skipped += 1
                continue
            full_text = row.get("full_text", "")
            if not full_text:
                continue
            for p in split_into_paragraphs(full_text, min_words=args.min_para_words):
                documents.append((p, ts))

    print(f"  Total paragraphs: {len(documents)}, skipped papers: {skipped}")
    ts_counts = Counter(ts for _, ts in documents)
    for t in sorted(ts_counts):
        lo, hi = YEAR_BINS[t]
        print(f"  T{t} ({lo}-{hi}): {ts_counts[t]} docs")

    # ── Step 2: Tokenize ─────────────────────────────────────────────────────
    print("\nStep 2: Tokenizing...")
    raw_texts = [d[0] for d in documents]
    tokenized = tokenize_spacy(raw_texts)
    processed = [" ".join(toks) for toks in tokenized]

    valid_idx = [i for i, t in enumerate(processed) if t.strip()]
    processed = [processed[i] for i in valid_idx]
    timestamps = [documents[i][1] for i in valid_idx]
    original_texts = [documents[i][0] for i in valid_idx]
    print(f"  Documents after filter: {len(processed)}")

    # ── Step 3: Build BoW ─────────────────────────────────────────────────────
    print(f"\nStep 3: Building BoW (min_df={args.min_df}, max_df={args.max_df})...")
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        min_df=args.min_df, max_df=args.max_df,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    bow = vectorizer.fit_transform(processed)
    vocab = vectorizer.get_feature_names_out()
    print(f"  BoW shape: {bow.shape}, vocab: {len(vocab)}")

    # Remove zero-word documents
    row_sums = np.array(bow.sum(axis=1)).flatten()
    nonzero = np.where(row_sums > 0)[0]
    bow = bow[nonzero]
    timestamps = [timestamps[i] for i in nonzero]
    original_texts = [original_texts[i] for i in nonzero]
    print(f"  After vocab filter: {bow.shape[0]} docs")

    # ── Step 4: Train/Val/Test split ──────────────────────────────────────────
    # Paper Section 4.3: "80% training set, 10% validation set, 10% test set"
    print(f"\nStep 4: Train/val/test split ({1-args.test_size-args.val_size:.0%}/{args.val_size:.0%}/{args.test_size:.0%})...")
    from sklearn.model_selection import train_test_split

    indices = np.arange(bow.shape[0])

    # First split: train+val vs test
    trainval_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=timestamps
    )

    # Second split: train vs val (from trainval)
    trainval_timestamps = [timestamps[i] for i in trainval_idx]
    val_fraction = args.val_size / (1.0 - args.test_size)  # e.g. 0.1/0.9 ≈ 0.111
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_fraction, random_state=42, stratify=trainval_timestamps
    )

    train_bow = bow[train_idx]
    val_bow = bow[val_idx]
    test_bow = bow[test_idx]
    train_times = [timestamps[i] for i in train_idx]
    val_times = [timestamps[i] for i in val_idx]
    test_times = [timestamps[i] for i in test_idx]
    train_texts = [original_texts[i] for i in train_idx]
    val_texts = [original_texts[i] for i in val_idx]
    test_texts = [original_texts[i] for i in test_idx]
    print(f"  Train: {train_bow.shape[0]}, Val: {val_bow.shape[0]}, Test: {test_bow.shape[0]}")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    print(f"\nStep 5: Saving to {args.output_dir}/...")
    from scipy import sparse

    sparse.save_npz(os.path.join(args.output_dir, "train_bow.npz"), train_bow)
    sparse.save_npz(os.path.join(args.output_dir, "val_bow.npz"), val_bow)
    sparse.save_npz(os.path.join(args.output_dir, "test_bow.npz"), test_bow)

    with open(os.path.join(args.output_dir, "vocab.txt"), "w") as f:
        f.writelines(w + "\n" for w in vocab)

    with open(os.path.join(args.output_dir, "train_times.txt"), "w") as f:
        f.writelines(str(t) + "\n" for t in train_times)

    with open(os.path.join(args.output_dir, "val_times.txt"), "w") as f:
        f.writelines(str(t) + "\n" for t in val_times)

    with open(os.path.join(args.output_dir, "test_times.txt"), "w") as f:
        f.writelines(str(t) + "\n" for t in test_times)

    with open(os.path.join(args.output_dir, "train_texts.txt"), "w") as f:
        f.writelines(t.replace("\n", " ") + "\n" for t in train_texts)

    with open(os.path.join(args.output_dir, "val_texts.txt"), "w") as f:
        f.writelines(t.replace("\n", " ") + "\n" for t in val_texts)

    with open(os.path.join(args.output_dir, "test_texts.txt"), "w") as f:
        f.writelines(t.replace("\n", " ") + "\n" for t in test_texts)

    with open(os.path.join(args.output_dir, "time2id.txt"), "w") as f:
        for i, (lo, hi) in enumerate(YEAR_BINS):
            f.write(f"{i}\t{lo}-{hi}\n")

    np.savetxt(os.path.join(args.output_dir, "train_idx.txt"), train_idx, fmt="%d")
    np.savetxt(os.path.join(args.output_dir, "val_idx.txt"), val_idx, fmt="%d")
    np.savetxt(os.path.join(args.output_dir, "test_idx.txt"), test_idx, fmt="%d")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Documents: {bow.shape[0]}  |  Vocab: {len(vocab)}  |  Timestamps: {len(set(timestamps))}")
    print(f"  Train: {train_bow.shape[0]}  |  Val: {val_bow.shape[0]}  |  Test: {test_bow.shape[0]}")


if __name__ == "__main__":
    main()
