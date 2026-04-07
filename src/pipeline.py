"""
CoNTM Pipeline — Continual Neural Topic Model (Algorithm 2).

This implements the main continual learning loop from the paper:

    Algorithm 2: CoNTM
    ─────────────────────────────────────────────
    Input: Timestamps D_1, ..., D_T ; K topics
    Output: Global memory ϕ_global ; Per-timestamp topics
    ─────────────────────────────────────────────
    1. Initialize ϕ_global ← 0  (K × V zeros)
    2. For t = 1, ..., T:
    3.     Initialize encoder parameters θ_t (fresh)
    4.     Initialize local delta Δϕ_local_t ← 0
    5.     For each epoch:
    6.         Train VAE on D_t using ELBO with β = ϕ_global + Δϕ_local_t
    7.     End for
    8.     ϕ_local_t = ϕ_global + Δϕ_local_t    (combined local logits)
    9.     ϕ_global ← (1 - ρ_t) · ϕ_global + ρ_t · ϕ_local_t
    10.    Store topics_t = top-m words from softmax(ϕ_local_t)
    11. End for
    ─────────────────────────────────────────────
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from .vae import ProdLDA, train_vae
from .data_loader import Corpus, make_dataloader, load_corpus, get_reference_corpus
from .global_memory import GlobalMemory
from .topic_utils import (
    extract_topics,
    compute_npmi,
    compute_topic_diversity,
    compute_topic_quality,
    compute_tts,
    compute_perplexity,
    print_topics,
    save_topics,
)


def run_contm(
    corpus: Corpus,
    n_topics: int = 50,
    enc_hidden: int = 256,
    dropout: float = 0.2,
    lr: float = 0.01,
    weight_decay: float = 1e-6,
    epochs: int = 100,
    kl_warmup_epochs: int = 20,
    patience: int = 10,
    batch_size: int = 256,
    kappa: float = 0.7,
    tau0: float = 1.0,
    top_m: int = 15,
    npmi_top_n: int = 10,
    output_dir: str = "outputs",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Run the full CoNTM pipeline (Algorithm 2).

    Args:
        corpus: loaded Corpus object with per-timestamp data
        n_topics: number of topics K
        enc_hidden: encoder hidden layer size
        dropout: encoder dropout rate
        lr: learning rate (paper uses 0.01)
        weight_decay: L2 regularization weight
        epochs: max training epochs per timestamp
        kl_warmup_epochs: epochs for KL annealing
        patience: early stopping patience
        batch_size: mini-batch size
        kappa: κ for step size decay
        tau0: τ₀ for step size offset
        top_m: number of top words per topic
        npmi_top_n: number of words for NPMI calculation
        output_dir: directory to save results
        device: "cpu" or "cuda"
        seed: random seed

    Returns:
        dict with per-timestamp results and aggregated metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamps = corpus.timestamps
    T = len(timestamps)
    vocab = corpus.vocab
    vocab_size = corpus.vocab_size

    print(f"\n{'=' * 70}")
    print(f" CoNTM: {T} timestamps, {n_topics} topics, V={vocab_size}")
    print(f" Hyperparams: κ={kappa}, τ₀={tau0}, lr={lr}, hidden={enc_hidden}")
    print(f" Device: {device}")
    print(f"{'=' * 70}\n")

    # ── Initialize Global Memory (Algorithm 2, line 1) ──
    global_memory = GlobalMemory(
        n_topics=n_topics,
        vocab_size=vocab_size,
        kappa=kappa,
        tau0=tau0,
    )

    # ── Storage ──
    results_per_ts = {}
    all_beta_dists = []      # for TTS computation
    all_topics_per_ts = {}   # for final output

    for t_idx, ts in enumerate(timestamps):
        t_num = t_idx + 1  # 1-based timestamp index
        ts_label = corpus.time_labels.get(ts, str(ts))

        print(f"\n{'─' * 60}")
        print(f" Timestamp {t_num}/{T}: {ts_label} "
              f"({corpus.train_data[ts].n_docs} train docs)")
        print(f"{'─' * 60}")

        # ── Prepare DataLoaders ──
        train_loader = make_dataloader(
            corpus.train_data[ts], batch_size=batch_size, shuffle=True
        )
        test_loader = None
        if corpus.test_data and ts in corpus.test_data:
            test_loader = make_dataloader(
                corpus.test_data[ts], batch_size=batch_size, shuffle=False
            )

        # ── Get global beta logits for this timestamp (frozen during training) ──
        if t_num == 1:
            # First timestamp: no global memory yet, train from scratch
            global_beta_tensor = None
        else:
            gb = global_memory.get_global_beta_logits()
            global_beta_tensor = torch.from_numpy(gb).float()

        # ── Initialize fresh VAE (Algorithm 2, lines 3-4) ──
        model = ProdLDA(
            vocab_size=vocab_size,
            n_topics=n_topics,
            enc_hidden=enc_hidden,
            dropout=dropout,
        )

        # ── Train VAE (Algorithm 2, lines 5-7) ──
        t_start = time.time()
        train_result = train_vae(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            global_beta=global_beta_tensor,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            kl_warmup_epochs=kl_warmup_epochs,
            patience=patience,
            device=device,
        )
        train_time = time.time() - t_start

        # ── Get local topic-word logits ϕ_local_t (Algorithm 2, line 8) ──
        # ϕ_local_t = ϕ_global + Δϕ_local_t  (computed inside model)
        local_beta_logits = model.get_topic_word_logits(global_beta_tensor)  # (K, V)
        local_beta_dist = model.get_topic_word_dist(global_beta_tensor)      # (K, V) softmax

        # ── Update global memory (Algorithm 2, line 9) ──
        rho = global_memory.update(local_beta_logits, t=t_num)

        # ── Extract topics (Algorithm 2, line 10) ──
        topics = extract_topics(local_beta_dist, vocab, top_m=top_m, source=f"T{ts}")
        all_topics_per_ts[ts] = topics
        all_beta_dists.append(local_beta_dist)

        # ── Evaluate: NPMI ──
        ref_docs = get_reference_corpus(corpus, ts, window=-1)
        tc = compute_npmi(topics, ref_docs, top_n=npmi_top_n)

        # ── Evaluate: Topic Diversity ──
        td = compute_topic_diversity(topics, top_n=top_m)

        # ── Evaluate: Topic Quality ──
        tq = compute_topic_quality(tc, td, n_topics, n_topics)

        # ── Evaluate: Predictive Perplexity (predict next timestamp) ──
        ppl = None
        if t_idx + 1 < T:
            next_ts = timestamps[t_idx + 1]
            if corpus.test_data and next_ts in corpus.test_data:
                next_test_loader = make_dataloader(
                    corpus.test_data[next_ts], batch_size=batch_size, shuffle=False
                )
                ppl = compute_perplexity(model, next_test_loader, global_beta_tensor, device)

        # ── Print summary ──
        print(f"\n  Results for T{ts} ({ts_label}):")
        print(f"    TC (NPMI):   {tc:.4f}")
        print(f"    TD:          {td:.4f}")
        print(f"    TQ:          {tq:.4f}")
        if ppl is not None:
            print(f"    PPL (next):  {ppl:.2f}")
        print(f"    ρ_t:         {rho:.4f}")
        print(f"    Train time:  {train_time:.1f}s")

        print_topics(topics[:10], label=f"T{ts} ({ts_label})")

        # ── Save per-timestamp outputs ──
        ts_dir = out / f"T{ts}"
        ts_dir.mkdir(parents=True, exist_ok=True)
        save_topics(topics, str(ts_dir / "topics.txt"), top_n=top_m)
        np.save(str(ts_dir / "beta_logits.npy"), local_beta_logits)
        np.save(str(ts_dir / "beta_dist.npy"), local_beta_dist)

        results_per_ts[ts] = {
            "timestamp": ts,
            "label": ts_label,
            "n_docs": corpus.train_data[ts].n_docs,
            "tc": tc,
            "td": td,
            "tq": tq,
            "ppl": ppl,
            "rho": rho,
            "train_time": train_time,
            "best_loss": train_result["best_loss"],
            "final_epoch": train_result["final_epoch"],
        }

    # ── Compute TTS (temporal topic smoothness) ──
    tts = compute_tts(all_beta_dists)

    # ── Aggregate metrics ──
    tc_values = [r["tc"] for r in results_per_ts.values()]
    td_values = [r["td"] for r in results_per_ts.values()]
    tq_values = [r["tq"] for r in results_per_ts.values()]
    ppl_values = [r["ppl"] for r in results_per_ts.values() if r["ppl"] is not None]

    summary = {
        "n_timestamps": T,
        "n_topics": n_topics,
        "vocab_size": vocab_size,
        "kappa": kappa,
        "tau0": tau0,
        "avg_tc": float(np.mean(tc_values)),
        "avg_td": float(np.mean(td_values)),
        "avg_tq": float(np.mean(tq_values)),
        "tts": tts,
        "avg_ppl": float(np.mean(ppl_values)) if ppl_values else None,
        "per_timestamp": {str(k): v for k, v in results_per_ts.items()},
    }

    # ── Print final summary ──
    print(f"\n{'=' * 70}")
    print(f" Final Summary")
    print(f"{'=' * 70}")
    print(f"  Avg TC (NPMI):  {summary['avg_tc']:.4f}")
    print(f"  Avg TD:         {summary['avg_td']:.4f}")
    print(f"  Avg TQ:         {summary['avg_tq']:.4f}")
    print(f"  TTS:            {summary['tts']:.4f}")
    if summary["avg_ppl"] is not None:
        print(f"  Avg PPL:        {summary['avg_ppl']:.2f}")
    print()

    # ── Save global memory and summary ──
    global_memory.save(str(out / "final_global_memory"))

    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save final global topics
    global_beta_dist = global_memory.get_global_beta_dist()
    global_topics = extract_topics(global_beta_dist, vocab, top_m=top_m, source="global")
    save_topics(global_topics, str(out / "final_global_topics.txt"), top_n=top_m)
    print_topics(global_topics[:10], label="Final Global")

    return summary
