"""
Topic evaluation utilities for CoNTM baseline.

Implements the evaluation metrics from the paper (Section 4.1), aligned with
SubNTM (fisherman611/SubNTM) evaluation methodology:

  - Topic Coherence (TC): NPMI via gensim CoherenceModel with temporal reference corpus
  - Topic Diversity (TD): 1 - TR (Burkhardt & Kramer, 2019) per-topic redundancy
  - Topic Quality (TQ): (1/K_ts) Sigma TC_i x TD_i x (T_i / T_max_i)
  - Temporal Topic Smoothness (TTS): Jaccard-based smoothness across consecutive timestamps
  - Predictive Perplexity (PPL): reconstruction perplexity on next timestamp's test set
  - IRBO: Inverted Rank-Biased Overlap for topic diversity (from SubNTM)

References:
  - Bouma (2009): NPMI
  - Burkhardt & Kramer (2019): Topic Redundancy (TR)
  - Karakkaparambil James et al. (2024): TTS
  - SubNTM (fisherman611/SubNTM): gensim coherence, IRBO
"""

import numpy as np
import scipy.sparse as sp
import itertools
from dataclasses import dataclass, field
from typing import Optional


# --- Topic Representation ---

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
        beta: (n_topics, vocab_size) topic-word distribution (each row sums to ~1)
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


# --- Topic Coherence (TC) --- NPMI via gensim ---

def compute_npmi(
    topics: list[Topic],
    reference_docs: list[list[str]],
    top_n: int = 10,
) -> float:
    """Compute TC using gensim's CoherenceModel with 'c_npmi'.

    Following SubNTM's evaluation approach:
        - Uses gensim.CoherenceModel for standardized NPMI computation
        - Reference corpus is the temporal reference (all docs up to current timestamp)

    From the paper (Section 4.1):
        "A temporal reference corpus was utilized to determine the NPMI score
        for the topics. This means that the NPMI score for a topic at a specific
        timestamp is calculated using the reference corpus available up to that
        point in time."

    Args:
        topics: list of Topic objects
        reference_docs: list of tokenized documents (each is a list of words)
        top_n: number of top words to use for NPMI computation

    Returns:
        Average NPMI across all topics.
    """
    if len(reference_docs) == 0:
        return 0.0

    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel

        # Build topic word lists
        topic_words = [t.words[:top_n] for t in topics]

        # Build gensim dictionary from the vocabulary present in reference docs
        dictionary = Dictionary(reference_docs)

        cm = CoherenceModel(
            texts=reference_docs,
            dictionary=dictionary,
            topics=topic_words,
            topn=top_n,
            coherence='c_npmi',
        )
        per_topic = cm.get_coherence_per_topic()
        return float(np.mean(per_topic))

    except ImportError:
        # Fallback to hand-rolled NPMI if gensim is not available
        return _compute_npmi_manual(topics, reference_docs, top_n)


def _compute_npmi_manual(
    topics: list[Topic],
    reference_docs: list[list[str]],
    top_n: int = 10,
) -> float:
    """Manual NPMI computation (fallback when gensim is not available)."""
    if len(reference_docs) == 0:
        return 0.0

    n_docs = len(reference_docs)

    word_doc_sets: dict[str, set[int]] = {}
    for doc_idx, doc in enumerate(reference_docs):
        for word in set(doc):
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
                    p_ij = df_ij / n_docs
                    p_i = df_i / n_docs
                    p_j = df_j / n_docs
                    pmi = np.log(p_ij / (p_i * p_j) + 1e-12)
                    npmi = pmi / (-np.log(p_ij) + 1e-12)
                    pairs_npmi.append(float(npmi))

        if pairs_npmi:
            coherences.append(np.mean(pairs_npmi))

    return float(np.mean(coherences)) if coherences else 0.0


# --- Topic Diversity (TD) --- Burkhardt & Kramer (2019) ---

def compute_topic_diversity(topics: list[Topic], top_n: int = 15) -> float:
    """Topic Diversity (TD) = 1 - Topic Redundancy (TR).

    From the paper (Section 4.1), using Burkhardt & Kramer (2019):
        TR(k) = 1/(K-1) x Sum_{i=1}^{N} Sum_{j!=k} P(w_ik, j)
    where P(w_ik, j) = 1 if word i of topic k appears in topic j's top words.

        TD = 1 - (1/K) Sum_k TR(k)

    This measures how much the top words of each topic overlap with other topics.
    Higher TD means less redundancy.

    Args:
        topics: list of Topic objects
        top_n: number of top words per topic to consider
    """
    if len(topics) < 2:
        return 1.0

    K = len(topics)
    topic_word_sets = [set(t.words[:top_n]) for t in topics]

    tr_per_topic = []
    for k in range(K):
        words_k = topic_word_sets[k]
        n_words = len(words_k)
        if n_words == 0:
            tr_per_topic.append(0.0)
            continue

        # Count how many of topic k's words appear in other topics
        overlap_count = 0
        for w in words_k:
            for j in range(K):
                if j != k and w in topic_word_sets[j]:
                    overlap_count += 1

        # TR(k) = overlap_count / (N x (K-1))
        tr_k = overlap_count / (n_words * (K - 1))
        tr_per_topic.append(tr_k)

    avg_tr = np.mean(tr_per_topic)
    td = 1.0 - avg_tr
    return float(td)


# --- IRBO --- Inverted Rank-Biased Overlap (from SubNTM) ---

def _rbo_score(list1: list[str], list2: list[str], p: float = 0.9) -> float:
    """Rank-Biased Overlap (RBO) extrapolated estimate between two ranked lists.

    Simplified implementation following SubNTM/evaluations/irbo.py.
    """
    k = min(len(list1), len(list2))
    if k == 0:
        return 0.0

    sum_val = 0.0
    for d in range(1, k + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        intersection = len(set1 & set2)
        total = len(set1) + len(set2)
        agreement = 2 * intersection / total if total > 0 else 0
        sum_val += p ** d * agreement

    return (1 - p) * sum_val


def compute_irbo(topics: list[Topic], top_n: int = 15, weight: float = 0.9) -> float:
    """Inverted Rank-Biased Overlap (IRBO) for topic diversity.

    From SubNTM: IRBO = 1 - mean(RBO(topic_i, topic_j)) for all pairs.
    Higher IRBO means more diverse topics.

    Args:
        topics: list of Topic objects
        top_n: number of top words per topic
        weight: RBO weight parameter p (default 0.9)
    """
    if len(topics) < 2:
        return 1.0

    top_word_lists = [t.words[:top_n] for t in topics]
    scores = []
    for list1, list2 in itertools.combinations(top_word_lists, 2):
        rbo_val = _rbo_score(list1, list2, p=weight)
        scores.append(rbo_val)

    return 1.0 - float(np.mean(scores))


# --- Topic Quality (TQ) ---

def compute_topic_quality(
    tc: float,
    td: float,
    n_topics: int,
    max_topics: int,
) -> float:
    """Topic Quality (TQ) = TC x TD x (T / T_max).

    From the paper (Section 4.1):
        TQ_t = TC_t x TD_t x (T_t / T_max_t)
    where T_t is the number of topics at timestamp t
    and T_max_t is the maximum number of topics across all timestamps.

    The paper's overall TQ is the average over all timestamps:
        TQ = (1/k) sum_{i=0}^{k-1} TQ_i

    Args:
        tc: topic coherence (NPMI) at this timestamp
        td: topic diversity at this timestamp
        n_topics: number of topics at this timestamp
        max_topics: maximum topics (T_max)
    """
    return tc * td * (n_topics / max_topics)


def compute_topic_quality_paper(
    tc_values: list[float],
    td_values: list[float],
    n_topics: int,
    max_topics: int,
) -> float:
    """Paper's overall TQ formula (Section 4.1):
        TQ = (1/k) sum_{i=0}^{k-1} TC_i x TD_i x (T_i / T_max_i)

    This is equivalent to averaging per-timestamp TQ when T_i = max_topics
    for all timestamps (as in our case with K=50 fixed).
    """
    if not tc_values:
        return 0.0
    k = len(tc_values)
    ratio = n_topics / max_topics
    return sum(tc * td * ratio for tc, td in zip(tc_values, td_values)) / k


# --- Temporal Topic Smoothness (TTS) ---

def compute_tts(
    topics_per_ts: list[list[Topic]],
    top_n: int = 15,
) -> float:
    """Temporal Topic Smoothness (TTS) — Jaccard variant.

    Uses Jaccard similarity between matched topics across consecutive timestamps.
    This is our original implementation (not the exact paper formula).

    Args:
        topics_per_ts: list of topic lists, one per timestamp in chronological order.
        top_n: number of top words per topic to consider.

    Returns:
        Average Jaccard-based smoothness across consecutive timestamps.
    """
    if len(topics_per_ts) < 2:
        return 0.0

    smoothness_scores = []
    for t in range(len(topics_per_ts) - 1):
        topics_t = topics_per_ts[t]
        topics_next = topics_per_ts[t + 1]

        K = min(len(topics_t), len(topics_next))
        if K == 0:
            continue

        # Per-topic Jaccard similarity (matched by index k)
        jaccard_scores = []
        for k in range(K):
            set_t = set(topics_t[k].words[:top_n])
            set_next = set(topics_next[k].words[:top_n])
            intersection = len(set_t & set_next)
            union = len(set_t | set_next)
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)

        smoothness_scores.append(np.mean(jaccard_scores))

    return float(np.mean(smoothness_scores))


def compute_tts_paper(
    topics_per_ts: list[list[Topic]],
    top_n: int = 15,
) -> float:
    """Temporal Topic Smoothness (TTS) — Paper version.

    From Karakkaparambil James et al. (2024) "Evaluating Dynamic Topic Models":
        TTS(k, t, t+1) = |top_n(k,t) ∩ top_n(k,t+1)| / n

    Topics are matched by index. Final TTS is averaged over all topics and
    all consecutive timestamp pairs.

    This is the EXACT formula from the paper:
        TTS = (1/K) Σ_k (1/(T-1)) Σ_t overlap(k,t,t+1) / n

    Args:
        topics_per_ts: list of topic lists, one per timestamp in chronological order.
        top_n: number of top words per topic to consider.

    Returns:
        Average overlap-based smoothness in [0, 1].
    """
    if len(topics_per_ts) < 2:
        return 0.0

    T = len(topics_per_ts)
    K = min(len(ts_topics) for ts_topics in topics_per_ts)
    if K == 0:
        return 0.0

    # Per-topic smoothness averaged over time pairs
    per_topic_smoothness = []
    for k in range(K):
        pair_scores = []
        for t in range(T - 1):
            set_t = set(topics_per_ts[t][k].words[:top_n])
            set_next = set(topics_per_ts[t + 1][k].words[:top_n])
            overlap = len(set_t & set_next) / top_n
            pair_scores.append(overlap)
        per_topic_smoothness.append(np.mean(pair_scores))

    return float(np.mean(per_topic_smoothness))


# --- Predictive Perplexity (PPL) ---

def compute_perplexity(
    model,
    test_loader,
    global_beta=None,
    device: str = "cpu",
) -> float:
    """Predictive perplexity on held-out documents.

    From the paper (Section 4.1 / Appendix L):
        PPL = exp( -Sum_d log p(d) / Sum_d N_d )

    The model trained at timestamp t is used to predict documents at timestamp t+1.
    Lower perplexity is better.

    Args:
        model: trained ProdLDA model
        test_loader: DataLoader for test documents
        global_beta: optional global topic-word logits (post-update phi_global)
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

            # Normalize BoW
            x_norm = batch / (batch.sum(dim=1, keepdim=True) + 1e-12)

            # Encode
            mu, log_var = model.encoder(x_norm)
            theta = model.reparameterize(mu, log_var)

            # Decode to get log p(w|d)
            log_recon = model.decoder(theta, global_beta)

            # log p(d) = Sum_w count(w) x log p(w|d)
            log_probs = (batch * log_recon).sum(dim=1)
            word_counts = batch.sum(dim=1)

            total_log_likelihood += log_probs.sum().item()
            total_words += word_counts.sum().item()

    if total_words == 0:
        return float("inf")

    # PPL = exp(-total_log_likelihood / total_words)
    avg_nll = -total_log_likelihood / total_words
    perplexity = np.exp(min(avg_nll, 500))  # cap to avoid overflow
    return float(perplexity)


# --- Summary / IO ---

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
