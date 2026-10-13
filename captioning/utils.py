from __future__ import annotations

import json
import logging
import os
from datetime import datetime


def setup_logger(runs_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(runs_dir, exist_ok=True)
    log_path = os.path.join(runs_dir, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_status(runs_dir: str, status: str, extra: dict | None = None) -> None:
    os.makedirs(runs_dir, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    path = os.path.join(runs_dir, "status.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # also a one-line human-readable file for quick `cat`
    with open(os.path.join(runs_dir, "STATUS"), "w", encoding="utf-8") as f:
        f.write(f"{status} | pid={os.getpid()} | {payload['updated_at']}\n")


class HistoryTracker:
    """Tracks per-epoch metrics and writes JSON + plots under runs/."""

    def __init__(self, runs_dir: str):
        self.runs_dir = runs_dir
        self.path = os.path.join(runs_dir, "history.json")
        self.records: list[dict] = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []

    def append(self, record: dict) -> None:
        self.records.append(record)
        self.save()
        self.plot()

    def save(self) -> None:
        os.makedirs(self.runs_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def plot(self) -> None:
        if not self.records:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        epochs = [r["epoch"] for r in self.records]
        train_loss = [r.get("train_loss") for r in self.records]
        val_loss = [r.get("val_loss") for r in self.records]

        # Loss curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, marker="o", label="train_loss")
        if any(v is not None for v in val_loss):
            ax.plot(epochs, val_loss, marker="s", label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train / Val Loss vs Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.runs_dir, "train_val_loss.png"), dpi=150)
        plt.close(fig)

        # BLEU curves
        bleu_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
        if any(k in self.records[0] for k in bleu_keys):
            fig, ax = plt.subplots(figsize=(8, 5))
            for k in bleu_keys:
                vals = [r.get(k) for r in self.records]
                if any(v is not None for v in vals):
                    ax.plot(epochs, vals, marker="o", label=k)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title("BLEU Scores vs Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.runs_dir, "bleu_scores.png"), dpi=150)
            plt.close(fig)

        # Other metrics
        other_keys = [k for k in ("METEOR", "ROUGE-L", "CIDEr-D") if any(k in r for r in self.records)]
        if other_keys:
            fig, ax = plt.subplots(figsize=(8, 5))
            for k in other_keys:
                vals = [r.get(k) for r in self.records]
                ax.plot(epochs, vals, marker="o", label=k)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title("Other Metrics vs Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.runs_dir, "other_metrics.png"), dpi=150)
            plt.close(fig)

        # LR / TF ratio
        if any("tf_ratio" in r or "lr_decoder" in r for r in self.records):
            fig, ax1 = plt.subplots(figsize=(8, 5))
            tf = [r.get("tf_ratio") for r in self.records]
            lr = [r.get("lr_decoder") for r in self.records]
            if any(v is not None for v in tf):
                ax1.plot(epochs, tf, marker="o", color="tab:blue", label="tf_ratio")
                ax1.set_ylabel("Teacher forcing ratio", color="tab:blue")
            ax2 = ax1.twinx()
            if any(v is not None for v in lr):
                ax2.plot(epochs, lr, marker="s", color="tab:orange", label="lr_decoder")
                ax2.set_ylabel("Decoder LR", color="tab:orange")
            ax1.set_xlabel("Epoch")
            ax1.set_title("Scheduled Sampling & Learning Rate")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.runs_dir, "schedule.png"), dpi=150)
            plt.close(fig)
