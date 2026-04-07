"""
Topic evaluation utilities for CoNTM baseline.

Implements the evaluation metrics from the paper (Section 4.1):
  - Topic Coherence (TC): NPMI over top-m words (m=10), computed per-timestamp
  - Topic Diversity (TD): 1 - topic repetition rate
  - Topic Quality (TQ): TC × TD × (T / T_max), aggregated over timestamps
  - Temporal Topic Smoothness (TTS): avg similarity of topic sets across consecutive timestamps
  - Predictive Perplexity (PPL): reconstruction perplexity on the NEXT timestamp's test set

Also provides:
  - extract_topics: extract top-m words from topic-word distributions
  - Topic dataclass for representation
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import Optional


# ─── Topic Representation ────────────────────────────────────────────────────

@dataclass
class Topic:
    """A single topic represented by its top words."""
    id: int
    words: list[str]
    word_weights: list[float]
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def to_string(self) -> str:
        return ", ".join(self.words)


def extract_topics(
    beta: np.ndarray,
    vocab: list[str],
    top_m: int = 15,
    source: str = "",
) -> list[Topic]:
    """Extract topics from a topic-word distribution matrix.

    Args:
        beta: (n_topics, vocab_size) topic-word distribution (each row sums to 1)
        vocab: list of vocabulary words
        top_m: number of top words to extract per topic

    Returns:
        List of Topic objects with top-m words and weights.
    """
    n_topics = beta.shape[0]
    topics = []
    for k in range(n_topics):
        top_idx = np.argsort(beta[k])[::-1][:top_m]
        words = [vocab[i] for i in top_idx]
        weights = [float(beta[k, i]) for i in top_idx]
        topics.append(Topic(id=k, words=words, word_weights=weights, source=source))
    return topics


# ─── Topic Coherence (NPMI) ──────────────────────────────────────────────────

def compute_npmi(
    topics: list[Topic],
    reference_docs: list[list[str]],
    top_n: int = 10,
) -> float:
    """Compute Normalized Pointwise Mutual Information (NPMI) topic coherence.

    From the paper (Section 4.1):
        TC = (1/T) Σ_k NPMI(top-m words of topic k)
    NPMI is computed using the *temporal* reference corpus (documents at the
    same timestamp), following Lenz & Winker (2020).

    Args:
        topics: list of Topic objects
        reference_docs: list of tokenized documents (each is a list of words)
        top_n: number of top words to use for NPMI computation

    Returns:
        Average NPMI across all topics.
    """
    if len(reference_docs) == 0:
        return 0.0

    n_docs = len(reference_docs)

    # Build word-document occurrence sets for efficient lookup
    word_doc_sets: dict[str, set[int]] = {}
    for doc_idx, doc in enumerate(reference_docs):
        for word in set(doc):  # unique words per document
            if word not in word_doc_sets:
                word_doc_sets[word] = set()
            word_doc_sets[word].add(doc_idx)

    coherences = []
    for topic in topics:
        words = topic.words[:top_n]
        if len(words) < 2:
            continue

        pairs_npmi = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w_i, w_j = words[i], words[j]
                docs_i = word_doc_sets.get(w_i, set())
                docs_j = word_doc_sets.get(w_j, set())
                docs_ij = docs_i & docs_j

                df_i = len(docs_i)
                df_j = len(docs_j)
                df_ij = len(docs_ij)

                if df_ij == 0 or df_i == 0 or df_j == 0:
                    pairs_npmi.append(-1.0)
                else:
                    # PMI(w_i, w_j) = log[ P(w_i, w_j) / (P(w_i) * P(w_j)) ]
                    p_ij = df_ij / n_docs
                    p_i = df_i / n_docs
                    p_j = df_j / n_docs
                    pmi = np.log(p_ij / (p_i * p_j) + 1e-12)
                    # NPMI = PMI / -log(P(w_i, w_j))
                    npmi = pmi / (-np.log(p_ij) + 1e-12)
                    pairs_npmi.append(float(npmi))

        if pairs_npmi:
            coherences.append(np.mean(pairs_npmi))

    return float(np.mean(coherences)) if coherences else 0.0


def compute_npmi_from_bow(
    topics: list[Topic],
    bow_matrix: sp.spmatrix,
    vocab: list[str],
    top_n: int = 10,
) -> float:
    """Compute NPMI using a BoW matrix directly (more efficient for large corpora).

    Args:
        topics: list of Topic objects
        bow_matrix: (n_docs, vocab_size) sparse BoW matrix
        vocab: vocabulary list
        top_n: number of top words per topic
    """
    n_docs = bow_matrix.shape[0]
    binary = (bow_matrix > 0).astype(np.float32)
    if sp.issparse(binary):
        binary = binary.toarray()

    word2idx = {w: i for i, w in enumerate(vocab)}
    doc_freq = binary.sum(axis=0).flatten()

    coherences = []
    for topic in topics:
        words = topic.words[:top_n]
        indices = [word2idx[w] for w in words if w in word2idx]
        if len(indices) < 2:
            continue

        pairs_npmi = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                wi, wj = indices[i], indices[j]
                df_i = doc_freq[wi]
                df_j = doc_freq[wj]
                df_ij = (binary[:, wi] * binary[:, wj]).sum()

                if df_ij == 0 or df_i == 0 or df_j == 0:
                    pairs_npmi.append(-1.0)
                else:
                    p_ij = df_ij / n_docs
                    p_i = df_i / n_docs
                    p_j = df_j / n_docs
                    pmi = np.log(p_ij / (p_i * p_j) + 1e-12)
                    npmi = pmi / (-np.log(p_ij) + 1e-12)
                    pairs_npmi.append(float(npmi))

        if pairs_npmi:
            coherences.append(np.mean(pairs_npmi))

    return float(np.mean(coherences)) if coherences else 0.0


# ─── Topic Diversity ──────────────────────────────────────────────────────────

def compute_topic_diversity(topics: list[Topic], top_n: int = 15) -> float:
    """Topic Diversity (TD) = 1 - Topic Repetition Rate (TR).

    From the paper (Section 4.1):
        TR = 1 - |unique words in top-m across all topics| / (T × m)
        TD = 1 - TR = |unique words| / (T × m)

    Higher TD means less redundancy across topics.
    """
    all_words = []
    for t in topics:
        all_words.extend(t.words[:top_n])
    if len(all_words) == 0:
        return 0.0
    return len(set(all_words)) / len(all_words)


# ─── Topic Quality ────────────────────────────────────────────────────────────

def compute_topic_quality(
    tc: float,
    td: float,
    n_topics: int,
    max_topics: int,
) -> float:
    """Topic Quality (TQ) = TC × TD × (T / T_max).

    From the paper (Section 4.1):
        TQ_t = TC_t × TD_t × (T_t / T_max)
    where T_t is the number of non-trivial topics at timestamp t
    and T_max is the maximum possible (e.g. n_topics = 50).

    Args:
        tc: topic coherence (NPMI)
        td: topic diversity
        n_topics: number of non-trivial topics at this timestamp
        max_topics: maximum possible topics (T_max)
    """
    return tc * td * (n_topics / max_topics)


# ─── Temporal Topic Smoothness (TTS) ─────────────────────────────────────────

def compute_tts(
    beta_list: list[np.ndarray],
) -> float:
    """Temporal Topic Smoothness (TTS).

    From the paper (Section 4.1):
        TTS = (1/(T-1)) Σ_{t=1}^{T-1} avg_k cos_sim(β_k^t, β_k^{t+1})

    Measures how smoothly topics evolve across consecutive timestamps.
    Higher TTS means more stable/smooth topic evolution.

    Args:
        beta_list: list of (n_topics, vocab_size) topic-word distributions,
                   one per timestamp in chronological order.

    Returns:
        Average cosine similarity between matched topics across consecutive timestamps.
    """
    if len(beta_list) < 2:
        return 0.0

    smoothness_scores = []
    for t in range(len(beta_list) - 1):
        beta_t = beta_list[t]      # (K, V)
        beta_next = beta_list[t + 1]  # (K, V)

        # Normalize rows to unit vectors for cosine similarity
        norm_t = beta_t / (np.linalg.norm(beta_t, axis=1, keepdims=True) + 1e-12)
        norm_next = beta_next / (np.linalg.norm(beta_next, axis=1, keepdims=True) + 1e-12)

        # Per-topic cosine similarity (matched by index)
        cos_sim = (norm_t * norm_next).sum(axis=1)  # (K,)
        smoothness_scores.append(cos_sim.mean())

    return float(np.mean(smoothness_scores))


# ─── Predictive Perplexity ────────────────────────────────────────────────────

def compute_perplexity(
    model,
    test_loader,
    global_beta=None,
    device: str = "cpu",
) -> float:
    """Predictive perplexity on held-out documents.

    From the paper (Section 4.1):
        PPL = exp( -Σ_d log p(d) / Σ_d N_d )
    where p(d) is the reconstructed probability and N_d is the total word count.

    The model trained at timestamp t is used to predict documents at timestamp t+1.
    Lower perplexity is better.

    Args:
        model: trained ProdLDA model
        test_loader: DataLoader for test documents
        global_beta: optional global topic-word logits
        device: compute device

    Returns:
        Perplexity (float). Returns inf if computation fails.
    """
    import torch

    model.eval()
    model = model.to(device)
    if global_beta is not None:
        global_beta = global_beta.to(device)

    total_log_likelihood = 0.0
    total_words = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch, global_beta, kl_weight=0.0)  # only recon

            # Log-likelihood per document
            # Reconstruction gives log p(w|d) via log_softmax
            # We need p(d) = Π_w p(w|d)^count(w)
            x_norm = batch / (batch.sum(dim=1, keepdim=True) + 1e-12)
            # Forward pass to get log-probabilities
            mu, log_var = model.encoder(x_norm)
            theta = model.reparameterize(mu, log_var)
            log_recon = model.decoder(theta, global_beta)

            # log p(d) = Σ_w count(w) * log p(w|d)
            log_probs = (batch * log_recon).sum(dim=1)  # (batch_size,)
            word_counts = batch.sum(dim=1)  # (batch_size,)

            total_log_likelihood += log_probs.sum().item()
            total_words += word_counts.sum().item()

    if total_words == 0:
        return float("inf")

    # PPL = exp(-total_log_likelihood / total_words)
    avg_nll = -total_log_likelihood / total_words
    perplexity = np.exp(min(avg_nll, 500))  # cap to avoid overflow
    return float(perplexity)


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_topics(topics: list[Topic], label: str = "") -> None:
    """Pretty-print topics to stdout."""
    header = f"Topics ({label})" if label else "Topics"
    print(f"\n{'=' * 60}")
    print(f" {header}: {len(topics)} topics")
    print(f"{'=' * 60}")
    for t in topics:
        words_str = ", ".join(t.words[:10])
        print(f"  [{t.id:2d}] {words_str}")
    print()


def save_topics(topics: list[Topic], path: str, top_n: int = 15) -> None:
    """Save topics to a text file (one topic per line)."""
    with open(path, "w") as f:
        for t in topics:
            words = t.words[:top_n]
            f.write(" ".join(words) + "\n")
