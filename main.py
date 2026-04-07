"""
CoNTM Baseline — Continual Neural Topic Model

Entry point for running the continual topic model pipeline.

Usage:
    python main.py                              # use default config
    python main.py --config configs/nyt.yaml    # use custom config
    python main.py --n_topics 30 --kappa 0.5    # override specific params

Paper: "CoNTM: Continual Neural Topic Model" (Algorithm 2)
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data_loader import load_corpus
from src.pipeline import run_contm


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="CoNTM Baseline — Continual Neural Topic Model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    # Allow overriding any config key from command line
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_topics", type=int, default=None)
    parser.add_argument("--enc_hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--kl_warmup_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--tau0", type=float, default=None)
    parser.add_argument("--top_m", type=int, default=None)
    parser.add_argument("--npmi_top_n", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Load base config
    config = load_config(args.config)

    # Override with command-line arguments
    for key in [
        "data_dir", "n_topics", "enc_hidden", "dropout", "lr", "weight_decay",
        "epochs", "kl_warmup_epochs", "patience", "batch_size", "kappa", "tau0",
        "top_m", "npmi_top_n", "output_dir", "seed", "device",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Auto-detect device (validate CUDA actually works, not just available)
    if config.get("device", "auto") == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                config["device"] = "cuda"
            except RuntimeError:
                print("⚠ CUDA available but not functional (driver mismatch?), falling back to CPU")
                config["device"] = "cpu"
        else:
            config["device"] = "cpu"
    elif config["device"] == "cuda":
        try:
            torch.zeros(1).cuda()
        except RuntimeError as e:
            print(f"⚠ CUDA requested but failed: {e}\n  Falling back to CPU")
            config["device"] = "cpu"

    # Print config
    print("\n Configuration:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print()

    # Load data
    corpus = load_corpus(config["data_dir"])

    # Run pipeline
    results = run_contm(
        corpus=corpus,
        n_topics=config["n_topics"],
        enc_hidden=config["enc_hidden"],
        dropout=config["dropout"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        epochs=config["epochs"],
        kl_warmup_epochs=config["kl_warmup_epochs"],
        patience=config["patience"],
        batch_size=config["batch_size"],
        kappa=config["kappa"],
        tau0=config["tau0"],
        top_m=config["top_m"],
        npmi_top_n=config["npmi_top_n"],
        output_dir=config["output_dir"],
        device=config["device"],
        seed=config["seed"],
    )

    print(f"\n Done! Results saved to {config['output_dir']}/")


if __name__ == "__main__":
    main()
