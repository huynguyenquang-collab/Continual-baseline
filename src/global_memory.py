"""
Global Memory for Continual Topic Models (CoNTM).

Implements the running-average global memory update from Algorithm 2 (line 9):
    ϕ_global = (1 - ρ_t) · ϕ_global + ρ_t · ϕ_local_t

where ρ_t = 1 / (τ₀ + t)^κ  is the decaying step size.

Default hyperparameters from the paper (Section 4.2):
    κ ∈ (0.5, 1], default κ = 0.7
    τ₀ ≥ 0, default τ₀ = 1
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GlobalMemory:
    """Stores the global topic-word logits and manages the running-average update.

    Attributes:
        n_topics: number of topics K
        vocab_size: vocabulary size V
        beta_logits: (K, V) global topic-word logits ϕ_global
        kappa: κ — decay exponent (default 0.7)
        tau0: τ₀ — base offset for step size (default 1)
        current_step: number of updates applied so far (starts at 0)
        history: list of ρ_t values for each update
    """
    n_topics: int
    vocab_size: int
    beta_logits: Optional[np.ndarray] = None   # (K, V) ϕ_global
    kappa: float = 0.7
    tau0: float = 1.0
    current_step: int = 0
    history: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.beta_logits is None:
            # Initialize to zeros (uniform in probability space after softmax)
            self.beta_logits = np.zeros((self.n_topics, self.vocab_size), dtype=np.float32)

    def compute_rho(self, t: Optional[int] = None) -> float:
        """Compute the step size ρ_t = 1 / (τ₀ + t)^κ.

        Args:
            t: timestamp index (1-based). If None, uses current_step + 1.

        Returns:
            ρ_t ∈ (0, 1]
        """
        if t is None:
            t = self.current_step + 1
        return 1.0 / ((self.tau0 + t) ** self.kappa)

    def update(self, local_beta_logits: np.ndarray, t: Optional[int] = None) -> float:
        """Update global memory with running average (Algorithm 2, line 9).

        ϕ_global ← (1 - ρ_t) · ϕ_global + ρ_t · ϕ_local_t

        For the first timestamp (t=1), ϕ_global is simply set to ϕ_local_t
        (equivalent to ρ_1 = 1 when ϕ_global is initialized to zeros).

        Args:
            local_beta_logits: (K, V) local topic-word logits ϕ_local_t
            t: timestamp index (1-based). If None, auto-incremented.

        Returns:
            The ρ_t value used for this update.
        """
        self.current_step += 1
        rho = self.compute_rho(t if t is not None else self.current_step)
        self.history.append(rho)

        assert local_beta_logits.shape == (self.n_topics, self.vocab_size), \
            f"Shape mismatch: expected ({self.n_topics}, {self.vocab_size}), " \
            f"got {local_beta_logits.shape}"

        self.beta_logits = (1 - rho) * self.beta_logits + rho * local_beta_logits

        return rho

    def get_global_beta_logits(self) -> np.ndarray:
        """Return the current global topic-word logits (K, V)."""
        return self.beta_logits.copy()

    def get_global_beta_dist(self) -> np.ndarray:
        """Return softmax-normalized global topic-word distribution (K, V)."""
        from scipy.special import softmax
        return softmax(self.beta_logits, axis=1)

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, output_dir: str, label: str = "") -> None:
        """Save global memory state to disk.

        Saves:
            - beta_logits as .npy
            - metadata (step, kappa, tau0, history) as .json
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        suffix = f"_{label}" if label else ""
        np.save(str(out / f"global_beta_logits{suffix}.npy"), self.beta_logits)

        metadata = {
            "n_topics": self.n_topics,
            "vocab_size": self.vocab_size,
            "kappa": self.kappa,
            "tau0": self.tau0,
            "current_step": self.current_step,
            "rho_history": self.history,
        }
        with open(out / f"global_memory_meta{suffix}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Global memory saved to {out} (step={self.current_step})")

    @classmethod
    def load(cls, output_dir: str, label: str = "") -> "GlobalMemory":
        """Load global memory state from disk."""
        out = Path(output_dir)
        suffix = f"_{label}" if label else ""

        beta_logits = np.load(str(out / f"global_beta_logits{suffix}.npy"))
        with open(out / f"global_memory_meta{suffix}.json") as f:
            meta = json.load(f)

        gm = cls(
            n_topics=meta["n_topics"],
            vocab_size=meta["vocab_size"],
            beta_logits=beta_logits,
            kappa=meta["kappa"],
            tau0=meta["tau0"],
            current_step=meta["current_step"],
            history=meta.get("rho_history", []),
        )
        return gm

    def __repr__(self) -> str:
        return (f"GlobalMemory(K={self.n_topics}, V={self.vocab_size}, "
                f"step={self.current_step}, κ={self.kappa}, τ₀={self.tau0})")
