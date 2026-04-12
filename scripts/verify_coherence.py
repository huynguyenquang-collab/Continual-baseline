#!/usr/bin/env python3
"""
Multi-method topic coherence verification.

Compares our NPMI scores against several independent measures:
  1. Manual document-level NPMI (our current fallback)
  2. Gensim c_npmi (sliding window=110, standard in literature)
  3. Gensim c_v (combination of indirect cosine similarity, best correlates with human judgment)
  4. Gensim u_mass (log-conditional probability, corpus-intrinsic)
  5. Word2Vec embedding coherence (average pairwise cosine similarity of top words)

Usage:
    cd CoNTM_baseline && uv run python3 scripts/verify_coherence.py --output_dir outputs_k20
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_corpus, get_reference_corpus
from src.topic_utils import split_text_word


def load_topics_from_file(path: str) -> list[list[str]]:
    """Load topics from a topics.txt file (one topic per line, space-separated words)."""
    topics = []
    with open(path) as f:
        for line in f:
            words = line.strip().split()
            if words:
                topics.append(words)
    return topics


def manual_npmi_document_level(topics: list[list[str]], docs: list[list[str]], top_n: int = 10) -> tuple[float, list[float]]:
    """Our current manual NPMI: document-level co-occurrence."""
    n_docs = len(docs)
    word_doc_sets = defaultdict(set)
    for doc_idx, doc in enumerate(docs):
        for word in set(doc):
            word_doc_sets[word].add(doc_idx)

    per_topic = []
    for topic_words in topics:
        words = topic_words[:top_n]
        if len(words) < 2:
            per_topic.append(0.0)
            continue
        pairs_npmi = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w_i, w_j = words[i], words[j]
                df_i = len(word_doc_sets.get(w_i, set()))
                df_j = len(word_doc_sets.get(w_j, set()))
                df_ij = len(word_doc_sets.get(w_i, set()) & word_doc_sets.get(w_j, set()))
                if df_ij == 0 or df_i == 0 or df_j == 0:
                    pairs_npmi.append(-1.0)
                    continue
                p_i = df_i / n_docs
                p_j = df_j / n_docs
                p_ij = df_ij / n_docs
                pmi = math.log(p_ij / (p_i * p_j))
                npmi = pmi / (-math.log(p_ij))
                pairs_npmi.append(npmi)
        per_topic.append(float(np.mean(pairs_npmi)) if pairs_npmi else 0.0)
    return float(np.mean(per_topic)), per_topic


def manual_npmi_skip_missing(topics: list[list[str]], docs: list[list[str]], top_n: int = 10) -> tuple[float, list[float]]:
    """Manual NPMI but SKIP pairs with zero co-occurrence instead of penalizing -1."""
    n_docs = len(docs)
    word_doc_sets = defaultdict(set)
    for doc_idx, doc in enumerate(docs):
        for word in set(doc):
            word_doc_sets[word].add(doc_idx)

    per_topic = []
    for topic_words in topics:
        words = topic_words[:top_n]
        if len(words) < 2:
            per_topic.append(0.0)
            continue
        pairs_npmi = []
        n_zero = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w_i, w_j = words[i], words[j]
                df_i = len(word_doc_sets.get(w_i, set()))
                df_j = len(word_doc_sets.get(w_j, set()))
                df_ij = len(word_doc_sets.get(w_i, set()) & word_doc_sets.get(w_j, set()))
                if df_ij == 0 or df_i == 0 or df_j == 0:
                    n_zero += 1
                    continue  # skip instead of -1
                p_i = df_i / n_docs
                p_j = df_j / n_docs
                p_ij = df_ij / n_docs
                pmi = math.log(p_ij / (p_i * p_j))
                npmi = pmi / (-math.log(p_ij))
                pairs_npmi.append(npmi)
        score = float(np.mean(pairs_npmi)) if pairs_npmi else 0.0
        per_topic.append(score)
    return float(np.mean(per_topic)), per_topic


def topmost_npmi(
    topics: list[list[str]],
    docs: list[list[str]],
    vocab: list[str],
    top_n: int = 10,
    coherence_type: str = 'c_npmi',
) -> tuple[float, list[float]]:
    """Replicate TopMost's _coherence() from github.com/bobxwu/topmost.

    Key difference from gensim_coherence(): the gensim Dictionary is built from
    the *vocabulary list* (not from the reference documents). This ensures every
    topic word gets a valid ID even if it is rare in the reference corpus, which
    avoids KeyErrors and gives the same behaviour as the TopMost codebase.

    Reference:
        topmost/eva/topic_coherence.py  (BobXWu/TopMost, main branch)
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError:
        return float('nan'), []

    topic_words = [t[:top_n] for t in topics]

    # TopMost builds the dictionary from vocab words, each treated as a
    # single-token "document": Dictionary([[w] for w in vocab])
    dictionary = Dictionary([[w] for w in vocab])

    cm = CoherenceModel(
        texts=docs,
        dictionary=dictionary,
        topics=topic_words,
        topn=top_n,
        coherence=coherence_type,
    )
    per_topic = cm.get_coherence_per_topic()
    return float(np.mean(per_topic)), list(per_topic)


def topmost_dynamic_npmi(
    corpus,
    topics_by_ts: dict[int, list[list[str]]],
    vocab: list[str],
    top_n: int = 10,
    coherence_type: str = 'c_npmi',
) -> tuple[float, dict[int, float]]:
    """Replicate TopMost's dynamic_coherence() from github.com/bobxwu/topmost.

    Uses ONLY the documents of each individual timestamp as the reference
    corpus (not the cumulative window), mirroring the TopMost convention:
        "use the texts of each time slice as the reference corpus."
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError:
        return float('nan'), {}

    dictionary = Dictionary([[w] for w in vocab])
    per_ts: dict[int, float] = {}

    for ts, topics in topics_by_ts.items():
        # Reference corpus = only this timestamp's train docs
        if ts not in corpus.train_data:
            continue
        bow = corpus.train_data[ts].bow
        ts_docs: list[list[str]] = []
        for i in range(bow.shape[0]):
            row = bow[i].toarray().flatten()
            word_indices = row.nonzero()[0]
            tokens: list[str] = []
            for idx in word_indices:
                tokens.extend([vocab[idx]] * int(row[idx]))
            if tokens:
                ts_docs.append(tokens)

        if not ts_docs:
            continue

        topic_words = [t[:top_n] for t in topics]
        cm = CoherenceModel(
            texts=ts_docs,
            dictionary=dictionary,
            topics=topic_words,
            topn=top_n,
            coherence=coherence_type,
        )
        per_ts[ts] = float(np.mean(cm.get_coherence_per_topic()))

    avg = float(np.mean(list(per_ts.values()))) if per_ts else float('nan')
    return avg, per_ts


def gensim_coherence(topics: list[list[str]], docs: list[list[str]], vocab: list[str], top_n: int = 10, measure: str = 'c_npmi') -> tuple[float, list[float]]:
    """Compute coherence using gensim's CoherenceModel."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError:
        return float('nan'), []

    topic_words = [t[:top_n] for t in topics]
    split_reference_corpus = docs
    dictionary = Dictionary(split_text_word(vocab))
    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=topic_words,
        topn=top_n,
        coherence=measure,
    )
    per_topic = cm.get_coherence_per_topic()
    return float(np.mean(per_topic)), per_topic


def w2v_coherence(topics: list[list[str]], docs: list[list[str]], top_n: int = 10) -> tuple[float, list[float]]:
    """Train Word2Vec on the reference corpus, then measure avg pairwise cosine similarity of topic words."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        return float('nan'), []

    # Train a small Word2Vec model on the reference docs
    model = Word2Vec(sentences=docs, vector_size=100, window=10, min_count=2, workers=4, epochs=20, seed=42)

    per_topic = []
    for topic_words in topics:
        words = [w for w in topic_words[:top_n] if w in model.wv]
        if len(words) < 2:
            per_topic.append(0.0)
            continue
        sims = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                sims.append(model.wv.similarity(words[i], words[j]))
        per_topic.append(float(np.mean(sims)))
    return float(np.mean(per_topic)), per_topic


def count_zero_cooc_pairs(topics: list[list[str]], docs: list[list[str]], top_n: int = 10) -> dict:
    """Count how many word pairs in topics never co-occur in any document."""
    word_doc_sets = defaultdict(set)
    for doc_idx, doc in enumerate(docs):
        for word in set(doc):
            word_doc_sets[word].add(doc_idx)

    total_pairs = 0
    zero_pairs = 0
    oov_words = set()
    per_topic_zero = []
    for topic_words in topics:
        words = topic_words[:top_n]
        topic_zero = 0
        topic_total = 0
        for w in words:
            if w not in word_doc_sets:
                oov_words.add(w)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                topic_total += 1
                total_pairs += 1
                df_ij = len(word_doc_sets.get(words[i], set()) & word_doc_sets.get(words[j], set()))
                if df_ij == 0:
                    zero_pairs += 1
                    topic_zero += 1
        per_topic_zero.append(topic_zero / topic_total if topic_total else 0)

    return {
        "total_pairs": total_pairs,
        "zero_pairs": zero_pairs,
        "zero_pct": zero_pairs / total_pairs * 100 if total_pairs else 0,
        "oov_words": len(oov_words),
        "avg_zero_pct_per_topic": float(np.mean(per_topic_zero)) * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs_k20")
    parser.add_argument("--data_dir", default="data/NIPS")
    parser.add_argument("--timestamps", nargs="*", type=int, default=None,
                        help="Which timestamps to check (default: 0, 5, 10)")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    ts_list = args.timestamps or [0, 5, 10]

    print(f"{'='*80}")
    print(f" Multi-Method Coherence Verification")
    print(f" Output: {args.output_dir}, top_n={args.top_n}")
    print(f"{'='*80}\n")

    all_results = {}

    for ts in ts_list:
        ts_label = corpus.time_labels.get(ts, str(ts))
        topics_path = os.path.join(args.output_dir, f"T{ts}", "topics.txt")
        if not os.path.exists(topics_path):
            print(f"  [SKIP] T{ts}: no topics.txt found")
            continue

        topics = load_topics_from_file(topics_path)
        n_topics = len(topics)
        vocab = corpus.vocab  # full vocabulary list, as used by TopMost

        # Build reference corpus: all docs up to and including this timestamp
        ref_docs = get_reference_corpus(corpus, ts, window=ts)
        n_ref_docs = len(ref_docs)

        print(f"{'─'*70}")
        print(f" T{ts} ({ts_label}): {n_topics} topics, {n_ref_docs} reference docs")
        print(f"{'─'*70}")

        # Show first 3 topics for quick visual check
        print(f"\n  Sample topics (top {args.top_n} words):")
        for i, t in enumerate(topics[:3]):
            print(f"    Topic {i}: {' '.join(t[:args.top_n])}")
        print(f"    ... ({n_topics - 3} more)\n")

        # 0. Zero co-occurrence analysis
        zero_info = count_zero_cooc_pairs(topics, ref_docs, args.top_n)
        print(f"  Zero co-occurrence analysis:")
        print(f"    Total word pairs:        {zero_info['total_pairs']}")
        print(f"    Zero co-occurrence:      {zero_info['zero_pairs']} ({zero_info['zero_pct']:.1f}%)")
        print(f"    Avg zero % per topic:    {zero_info['avg_zero_pct_per_topic']:.1f}%")
        print(f"    OOV words:               {zero_info['oov_words']}")
        print()

        # 1. Our manual NPMI (document-level, -1 for zero co-occ)
        # NOTE: -1 penalty for zero-cooc pairs can make scores very negative when
        # many word pairs never appear in the same document. Use method 2 instead.
        m1_avg, m1_per = manual_npmi_document_level(topics, ref_docs, args.top_n)
        print(f"  1. Manual NPMI (doc-level, -1 penalty):     {m1_avg:.4f}  [WARNING: -1 penalty inflates negative values]")

        # 2. Manual NPMI (skip zero co-occurrence) — preferred document-level metric
        m2_avg, m2_per = manual_npmi_skip_missing(topics, ref_docs, args.top_n)
        print(f"  2. Manual NPMI (doc-level, skip zeros):     {m2_avg:.4f}  [PREFERRED: skips missing pairs]")

        # 3. Gensim c_npmi (sliding window)
        m3_avg, m3_per = gensim_coherence(topics, ref_docs, vocab, args.top_n, 'c_npmi')
        print(f"  3. Gensim c_npmi (sliding window=110):      {m3_avg:.4f}" if not math.isnan(m3_avg) else "  3. Gensim c_npmi: NOT INSTALLED")

        # 4. Gensim c_v (best human correlation)
        m4_avg, m4_per = gensim_coherence(topics, ref_docs, vocab, args.top_n, 'c_v')
        print(f"  4. Gensim c_v (best human corr):            {m4_avg:.4f}" if not math.isnan(m4_avg) else "  4. Gensim c_v: NOT INSTALLED")

        # 5. Gensim u_mass (corpus-intrinsic)
        m5_avg, m5_per = gensim_coherence(topics, ref_docs, vocab, args.top_n, 'u_mass')
        print(f"  5. Gensim u_mass (corpus-intrinsic):        {m5_avg:.4f}" if not math.isnan(m5_avg) else "  5. Gensim u_mass: NOT INSTALLED")

        # 6. Word2Vec embedding coherence
        m6_avg, m6_per = w2v_coherence(topics, ref_docs, args.top_n)
        print(f"  6. Word2Vec cosine coherence:               {m6_avg:.4f}" if not math.isnan(m6_avg) else "  6. Word2Vec: NOT INSTALLED")

        # 7. TopMost NPMI — cumulative reference corpus, dict built from vocab
        #    Matches github.com/bobxwu/topmost topmost/eva/topic_coherence.py
        m7_avg, m7_per = topmost_npmi(topics, ref_docs, vocab, args.top_n, 'c_npmi')
        print(f"  7. TopMost NPMI (vocab-dict, cumul. ref):   {m7_avg:.4f}" if not math.isnan(m7_avg) else "  7. TopMost NPMI: NOT INSTALLED")

        # 8. TopMost dynamic NPMI — per-timestamp reference corpus only
        #    Matches github.com/bobxwu/topmost topmost/eva/topic_coherence.py::dynamic_coherence
        m8_avg, m8_per_ts = topmost_dynamic_npmi(corpus, {ts: topics}, vocab, args.top_n, 'c_npmi')
        print(f"  8. TopMost dynamic NPMI (ts-only ref):      {m8_avg:.4f}" if not math.isnan(m8_avg) else "  8. TopMost dynamic NPMI: NOT INSTALLED")

        # Per-topic breakdown for the first 5 topics
        print(f"\n  Per-topic breakdown (first 5):")
        print(f"    {'Topic':>7} | {'ManNPMI':>8} | {'SkipZero':>8} | {'g_npmi':>8} | {'g_cv':>8} | {'g_umass':>8} | {'w2v':>8} | {'tm_npmi':>8}")
        print(f"    {'─'*7}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")
        for i in range(min(5, n_topics)):
            row = f"    {i:>7} | {m1_per[i]:>8.4f} | {m2_per[i]:>8.4f}"
            row += f" | {m3_per[i]:>8.4f}" if m3_per else f" | {'N/A':>8}"
            row += f" | {m4_per[i]:>8.4f}" if m4_per else f" | {'N/A':>8}"
            row += f" | {m5_per[i]:>8.4f}" if m5_per else f" | {'N/A':>8}"
            row += f" | {m6_per[i]:>8.4f}" if m6_per else f" | {'N/A':>8}"
            row += f" | {m7_per[i]:>8.4f}" if m7_per else f" | {'N/A':>8}"
            print(row)

        print()
        all_results[ts] = {
            "label": ts_label,
            "n_topics": n_topics,
            "n_ref_docs": n_ref_docs,
            "zero_cooc_pct": zero_info["zero_pct"],
            "manual_npmi": m1_avg,
            "manual_npmi_skip_zero": m2_avg,
            "gensim_c_npmi": m3_avg,
            "gensim_c_v": m4_avg,
            "gensim_u_mass": m5_avg,
            "w2v_cosine": m6_avg,
            "topmost_npmi": m7_avg,
            "topmost_dynamic_npmi": m8_avg,
        }

    # Final summary table
    print(f"\n{'='*80}")
    print(f" Summary")
    print(f"{'='*80}")
    print(f"  {'TS':>4} | {'Label':>12} | {'ManNPMI':>8} | {'SkipZero':>8} | {'g_npmi':>8} | {'g_cv':>8} | {'g_umass':>8} | {'w2v':>8} | {'tm_npmi':>8} | {'tm_dyn':>8} | {'Zero%':>6}")
    print(f"  {'─'*4}-+-{'─'*12}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*6}")
    for ts, r in sorted(all_results.items()):
        print(f"  {ts:>4} | {r['label']:>12} | {r['manual_npmi']:>8.4f} | {r['manual_npmi_skip_zero']:>8.4f}"
              f" | {r['gensim_c_npmi']:>8.4f} | {r['gensim_c_v']:>8.4f} | {r['gensim_u_mass']:>8.4f}"
              f" | {r['w2v_cosine']:>8.4f} | {r['topmost_npmi']:>8.4f} | {r['topmost_dynamic_npmi']:>8.4f}"
              f" | {r['zero_cooc_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
