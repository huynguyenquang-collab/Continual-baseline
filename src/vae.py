"""
ProdLDA-style VAE for topic modeling (CoNTM baseline).

Architecture (from the paper, Section 3.2):
  Encoder: BoW → FC → FC → μ, log(σ²) → Logistic-Normal (Laplace approx to Dirichlet)
  Decoder: θ (topic proportions) → β (topic-word matrix) → Reconstructed BoW

Residual delta learning (Algorithm 2, lines 6-9):
  At timestamp t, the decoder topic-word logits are:
    ϕ_local_t = ϕ_global + Δϕ_local_t
  where ϕ_global comes from the previous global state.
  The word probability is: p(w|z) = softmax(ϕ_local_t · z)

References:
  - Srivastava & Sutton, 2017 (ProdLDA / product of experts)
  - Burkhardt & Kramer, 2019 (DVAE topic model)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from dataclasses import dataclass


@dataclass
class VAEOutput:
    """Output from a single VAE forward pass."""
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    loss: torch.Tensor
    theta: torch.Tensor  # (batch, n_topics) topic proportions


class ProdLDAEncoder(nn.Module):
    """Encoder: BoW → (μ, log σ²) in logistic-normal space.

    Two hidden layers with softplus activation and batch normalization.
    Dropout is applied to the input (normalized BoW).
    """

    def __init__(self, vocab_size: int, n_topics: int, hidden: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.mu = nn.Linear(hidden, n_topics)
        self.log_var = nn.Linear(hidden, n_topics)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, vocab_size) normalized BoW
        Returns:
            mu: (batch, n_topics)
            log_var: (batch, n_topics)
        """
        h = self.drop(x)
        h = F.softplus(self.bn1(self.fc1(h)))
        h = F.softplus(self.bn2(self.fc2(h)))
        return self.mu(h), self.log_var(h)


class ProdLDADecoder(nn.Module):
    """Decoder: θ → reconstructed BoW via topic-word matrix β.

    From the paper (Section 3.2.2):
      p(w_n = w | z, ϕ_global, Δϕ_local_t) = [σ(g(ϕ_global, Δϕ_local_t) · z)]_w
    where σ is softmax and g(ϕ_global, Δϕ) = ϕ_global + Δϕ.

    The softmax is applied AFTER multiplying θ·β (product of experts).
    Uses BatchNorm on the pre-softmax logits for stability.

    IMPORTANT: Δϕ_local_t is initialized to ZERO (Algorithm 2, line 3).
    """

    def __init__(self, vocab_size: int, n_topics: int):
        super().__init__()
        # Local delta topic-word logits Δϕ_local_t
        self.topic_word_logits = nn.Linear(n_topics, vocab_size, bias=False)
        # Initialize Δϕ_local_t = 0 (Algorithm 2, line 3)
        nn.init.zeros_(self.topic_word_logits.weight)
        # BN on pre-softmax logits (critical for ProdLDA stability)
        self.bn = nn.BatchNorm1d(vocab_size, affine=True)

    def forward(self, theta: torch.Tensor, global_beta: Optional[torch.Tensor] = None):
        """
        Args:
            theta: (batch, n_topics) topic proportions (softmax output from encoder)
            global_beta: (n_topics, vocab_size) global topic-word logits ϕ_global (optional)
        Returns:
            log_recon: (batch, vocab_size) log-probability of words
        """
        # Local delta logits: Δϕ_local_t
        local_logits = self.topic_word_logits.weight.T  # (n_topics, vocab_size)

        # g(ϕ_global, Δϕ_local_t) = ϕ_global + Δϕ_local_t
        if global_beta is not None:
            combined = global_beta + local_logits
        else:
            combined = local_logits

        # θ · β_logits → (batch, vocab_size) unnormalized scores
        logits = torch.mm(theta, combined)  # (batch, vocab_size)
        logits = self.bn(logits)

        # Log-softmax for numerical stability (product of experts)
        log_recon = F.log_softmax(logits, dim=-1)
        return log_recon


class ProdLDA(nn.Module):
    """ProdLDA: Product of Experts LDA with logistic-normal prior.

    Implements the Neural Variational Document Model with Dirichlet-like prior
    via logistic-normal approximation (Srivastava & Sutton, 2017).
    """

    def __init__(self, vocab_size: int, n_topics: int, enc_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics

        self.encoder = ProdLDAEncoder(vocab_size, n_topics, enc_hidden, dropout)
        self.decoder = ProdLDADecoder(vocab_size, n_topics)

        # Prior parameters: standard normal N(0, I) as Dirichlet approximation
        self.prior_mu = nn.Parameter(torch.zeros(n_topics), requires_grad=False)
        self.prior_var = nn.Parameter(torch.ones(n_topics), requires_grad=False)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z ~ N(μ, σ²), then θ = softmax(z).

        The softmax maps the logistic-normal sample to the simplex,
        approximating a Dirichlet distribution (Section 3.2.2).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (batch, n_topics)
        theta = F.softmax(z, dim=-1)
        return theta

    def forward(self, x: torch.Tensor, global_beta: Optional[torch.Tensor] = None,
                kl_weight: float = 1.0) -> VAEOutput:
        """
        Args:
            x: (batch, vocab_size) raw BoW counts
            global_beta: optional (n_topics, vocab_size) global topic-word logits ϕ_global
            kl_weight: weight for KL divergence (for annealing during training)
        Returns:
            VAEOutput with reconstruction loss, KL loss, total loss, and θ
        """
        # Normalize BoW to get empirical word distribution
        x_norm = x / (x.sum(dim=1, keepdim=True) + 1e-12)

        # Encode: BoW → μ, log σ²
        mu, log_var = self.encoder(x_norm)

        # Sample: z ~ N(μ, σ²), θ = softmax(z)
        theta = self.reparameterize(mu, log_var)

        # Decode: θ → log p(w|d) via ϕ_global + Δϕ_local
        log_recon = self.decoder(theta, global_beta)

        # Reconstruction loss: -E[log p(w|z)] (negative log-likelihood)
        recon_loss = -(x * log_recon).sum(dim=1).mean()

        # KL divergence: KL(q(z|x) || p(z)) for logistic-normal
        prior_var = self.prior_var
        prior_mu = self.prior_mu
        var_ratio = torch.exp(log_var) / prior_var
        diff = mu - prior_mu
        kl = 0.5 * (var_ratio + diff.pow(2) / prior_var - 1.0 - log_var + prior_var.log())
        kl_loss = kl.sum(dim=1).mean()

        # ELBO = -recon_loss - kl_weight * kl_loss (we minimize the negative)
        loss = recon_loss + kl_weight * kl_loss

        return VAEOutput(
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            loss=loss,
            theta=theta,
        )

    def get_topic_word_dist(self, global_beta: Optional[torch.Tensor] = None) -> np.ndarray:
        """Get topic-word distributions β: (n_topics, vocab_size).

        Returns softmax-normalized probability distributions over words for each topic.
        These correspond to ϕ_local_t = softmax(ϕ_global + Δϕ_local_t).
        """
        with torch.no_grad():
            local_logits = self.decoder.topic_word_logits.weight.T.detach()
            if global_beta is not None:
                combined = global_beta.detach() + local_logits
            else:
                combined = local_logits
            beta = F.softmax(combined, dim=-1)
        return beta.cpu().numpy()

    def get_topic_word_logits(self, global_beta: Optional[torch.Tensor] = None) -> np.ndarray:
        """Get raw topic-word logits: (n_topics, vocab_size).

        These are the unnormalized ϕ_local_t = ϕ_global + Δϕ_local_t
        (useful for the global memory update in Algorithm 2, line 9).
        """
        with torch.no_grad():
            local_logits = self.decoder.topic_word_logits.weight.T.detach()
            if global_beta is not None:
                combined = global_beta.detach() + local_logits
            else:
                combined = local_logits
        return combined.cpu().numpy()


# ─── Training Loop ───────────────────────────────────────────────────────────

def _compute_val_ppl(model, val_loader, global_beta, device):
    """Compute perplexity on validation set (used for alpha-based early stopping).

    PPL = exp(-sum log p(d) / sum N_d)
    """
    model.eval()
    total_ll = 0.0
    total_words = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            x_norm = batch / (batch.sum(dim=1, keepdim=True) + 1e-12)
            mu, log_var = model.encoder(x_norm)
            theta = model.reparameterize(mu, log_var)
            log_recon = model.decoder(theta, global_beta)
            log_probs = (batch * log_recon).sum(dim=1)
            total_ll += log_probs.sum().item()
            total_words += batch.sum(dim=1).sum().item()
    if total_words == 0:
        return float("inf")
    avg_nll = -total_ll / total_words
    return float(np.exp(min(avg_nll, 500)))


def train_vae(
    model: ProdLDA,
    train_loader,
    val_loader=None,
    test_loader=None,
    global_beta: Optional[torch.Tensor] = None,
    epochs: int = 100,
    lr: float = 0.002,
    weight_decay: float = 1e-6,
    kl_warmup_epochs: int = 20,
    patience: int = 10,
    alpha: float = 0.9,
    device: str = "cpu",
) -> dict:
    """Train ProdLDA on a single timestamp's data (Algorithm 2, lines 5-8).

    At each timestamp, a fresh VAE is initialized and trained with the
    global beta logits frozen. Only the encoder θ and local delta Δϕ_local
    are optimized via gradient descent on the ELBO.

    Early stopping is based on the validation set (paper Section 4.3):
        monitor = alpha * val_loss + (1 - alpha) * val_ppl
    where alpha controls the trade-off between ELBO loss and predictive perplexity.
    (Appendix L, Figures 19/20)

    Args:
        model: ProdLDA instance (freshly initialized for this timestamp)
        train_loader: DataLoader yielding (batch, vocab_size) BoW tensors
        val_loader: DataLoader for validation/early stopping (paper: 10% of data)
        test_loader: DataLoader for test set (not used for early stopping)
        global_beta: (n_topics, vocab_size) frozen global logits from previous timestamps
        epochs: max training epochs
        lr: learning rate (paper uses 0.01)
        weight_decay: L2 regularization
        kl_warmup_epochs: number of epochs to linearly anneal KL weight from 0→1
        patience: early stopping patience
        alpha: trade-off for early stopping: alpha*val_loss + (1-alpha)*val_ppl
        device: "cpu" or "cuda"

    Returns:
        dict with training history and best loss
    """
    model = model.to(device)
    if global_beta is not None:
        global_beta = global_beta.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "train_recon": [], "train_kl": [],
               "val_loss": [], "test_loss": [], "kl_weight": []}

    for epoch in range(1, epochs + 1):
        # KL annealing: linearly increase KL weight from 0 to 1
        if kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / kl_warmup_epochs)
        else:
            kl_weight = 1.0
        history["kl_weight"].append(kl_weight)

        # ── Train ──
        model.train()
        total_loss = total_recon = total_kl = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch, global_beta, kl_weight)

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += output.loss.item()
            total_recon += output.recon_loss.item()
            total_kl += output.kl_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        history["train_loss"].append(avg_loss)
        history["train_recon"].append(avg_recon)
        history["train_kl"].append(avg_kl)

        # ── Validation (for early stopping, paper uses val set) ──
        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch, global_beta, kl_weight=1.0)
                    total_val += output.loss.item()
                    n_val += 1
            val_loss = total_val / n_val
            history["val_loss"].append(val_loss)
        else:
            history["val_loss"].append(None)

        # ── Test (separate, not for early stopping) ──
        test_loss = None
        if test_loader is not None:
            model.eval()
            total_test = 0.0
            n_test = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    output = model(batch, global_beta, kl_weight=1.0)
                    total_test += output.loss.item()
                    n_test += 1
            test_loss = total_test / n_test
            history["test_loss"].append(test_loss)
        else:
            history["test_loss"].append(None)

        # Monitor loss: use val if available, else train
        # Paper (Appendix L, Figures 19-20): alpha * val_loss + (1-alpha) * val_ppl
        if val_loader is not None and val_loss is not None:
            # Compute validation perplexity for alpha-weighted stopping
            val_ppl = _compute_val_ppl(model, val_loader, global_beta, device)
            monitor_loss = alpha * val_loss + (1 - alpha) * val_ppl
        else:
            monitor_loss = avg_loss
        scheduler.step(monitor_loss)

        # Early stopping
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1 or wait == 0:
            msg = f"  Epoch {epoch:3d} | Loss {avg_loss:.2f} (recon {avg_recon:.2f} + kl {avg_kl:.2f} * {kl_weight:.2f})"
            if val_loss is not None:
                msg += f" | Val {val_loss:.2f}"
            if test_loss is not None:
                msg += f" | Test {test_loss:.2f}"
            if wait == 0:
                msg += " *"
            print(msg)

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "history": history,
        "best_loss": best_loss,
        "final_epoch": epoch,
    }
