"""
Data loader for preprocessed BoW corpora with timestamp information.

Loads sparse BoW matrices, vocabulary, and timestamps.
Provides per-timestamp iterators for continual topic model training.

Expected data directory layout (after preprocessing):
    data/<dataset>/
        train_bow.npz       — sparse (n_train, V) BoW count matrix
        test_bow.npz        — sparse (n_test, V) BoW count matrix
        train_times.txt     — one integer timestamp per train document
        test_times.txt      — one integer timestamp per test document
        vocab.txt           — one word per line (V words)
        time2id.txt         — "id\tyear_range" mapping (e.g. "0\t1987-1989")
        train_texts.txt     — (optional) raw text per train document
        test_texts.txt      — (optional) raw text per test document
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class TimestampData:
    """Data for a single timestamp slice."""
    timestamp: int
    bow: sp.csr_matrix          # (n_docs, vocab_size) sparse BoW
    label: str = ""             # human-readable label, e.g. "1987-1989"
    n_docs: int = 0
    vocab_size: int = 0

    def __post_init__(self):
        self.n_docs, self.vocab_size = self.bow.shape


@dataclass
class Corpus:
    """Full corpus with train/test splits and per-timestamp access."""
    vocab: list[str]
    vocab_size: int
    timestamps: list[int]                      # sorted unique timestamp ids
    time_labels: dict[int, str]                # timestamp_id → human label
    train_data: dict[int, TimestampData]       # timestamp_id → TimestampData
    test_data: Optional[dict[int, TimestampData]] = None


class BoWDataset(Dataset):
    """PyTorch Dataset wrapping a sparse BoW matrix."""

    def __init__(self, bow: sp.csr_matrix):
        self.bow = bow
        self.n_docs, self.vocab_size = bow.shape

    def __len__(self) -> int:
        return self.n_docs

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.bow[idx].toarray().flatten().astype(np.float32)
        return torch.from_numpy(row)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_corpus(data_dir: str) -> Corpus:
    """Load a preprocessed corpus from disk.

    Args:
        data_dir: path to the dataset directory containing the expected files.

    Returns:
        Corpus with per-timestamp train (and optionally test) data.
    """
    data_dir = Path(data_dir)

    # Vocabulary
    vocab = (data_dir / "vocab.txt").read_text().strip().split("\n")
    vocab_size = len(vocab)

    # Time labels (optional mapping from id to human-readable string)
    time_labels: dict[int, str] = {}
    time2id_path = data_dir / "time2id.txt"
    if time2id_path.exists():
        for line in time2id_path.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                time_labels[int(parts[0])] = parts[1]

    # Train BoW + timestamps
    train_bow = sp.load_npz(str(data_dir / "train_bow.npz"))
    train_times = np.loadtxt(str(data_dir / "train_times.txt"), dtype=int)

    # Test BoW + timestamps (optional)
    test_bow_path = data_dir / "test_bow.npz"
    test_times_path = data_dir / "test_times.txt"
    has_test = test_bow_path.exists() and test_times_path.exists()
    if has_test:
        test_bow = sp.load_npz(str(test_bow_path))
        test_times = np.loadtxt(str(test_times_path), dtype=int)

    # Group documents by timestamp
    unique_ts = sorted(set(train_times.tolist()))

    train_data: dict[int, TimestampData] = {}
    for ts in unique_ts:
        mask = train_times == ts
        train_data[ts] = TimestampData(
            timestamp=ts,
            bow=train_bow[mask],
            label=time_labels.get(ts, str(ts)),
        )

    test_data: Optional[dict[int, TimestampData]] = None
    if has_test:
        test_data = {}
        for ts in unique_ts:
            mask = test_times == ts
            if mask.sum() > 0:
                test_data[ts] = TimestampData(
                    timestamp=ts,
                    bow=test_bow[mask],
                    label=time_labels.get(ts, str(ts)),
                )

    print(f"Loaded corpus: {vocab_size} vocab, {len(unique_ts)} timestamps, "
          f"{train_bow.shape[0]} train docs"
          + (f", {test_bow.shape[0]} test docs" if has_test else ""))

    return Corpus(
        vocab=vocab,
        vocab_size=vocab_size,
        timestamps=unique_ts,
        time_labels=time_labels,
        train_data=train_data,
        test_data=test_data,
    )


def make_dataloader(
    ts_data: TimestampData,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for a single timestamp's BoW data."""
    dataset = BoWDataset(ts_data.bow)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def get_reference_corpus(
    corpus: Corpus,
    timestamp: int,
    window: int = -1,
) -> list[list[str]]:
    """Build a reference corpus of tokenized documents for NPMI computation.

    From the paper (Section 4.1): "we compute TC with the set of documents
    belonging to a specific timestamp." If window > 0, include documents from
    [timestamp - window, timestamp].

    Args:
        corpus: the full Corpus object
        timestamp: current timestamp id
        window: how many past timestamps to include (-1 = only current timestamp)

    Returns:
        List of tokenized documents (each doc is a list of word strings).
    """
    # Collect relevant timestamps
    if window < 0:
        relevant_ts = [timestamp]
    else:
        relevant_ts = [t for t in corpus.timestamps
                       if (timestamp - window) <= t <= timestamp]

    docs: list[list[str]] = []
    vocab = corpus.vocab
    for ts in relevant_ts:
        if ts in corpus.train_data:
            bow = corpus.train_data[ts].bow
            # Convert sparse BoW back to token lists
            for i in range(bow.shape[0]):
                row = bow[i].toarray().flatten()
                word_indices = row.nonzero()[0]
                # Repeat each word by its count for NPMI
                tokens = []
                for idx in word_indices:
                    tokens.extend([vocab[idx]] * int(row[idx]))
                if tokens:
                    docs.append(tokens)
    return docs
