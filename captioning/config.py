from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # paths
    data_root: str = "data/coco"
    train_images: str = "data/coco/train2017"
    val_images: str = "data/coco/val2017"
    train_ann: str = "data/coco/annotations/captions_train2017.json"
    val_ann: str = "data/coco/annotations/captions_val2017.json"
    cache_dir: str = "data/coco/cache"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs/predictions"
    runs_dir: str = "runs"

    # vocab
    min_word_freq: int = 5
    max_caption_len: int = 50

    # model
    encoder_name: str = "efficientnet_b4"
    encoder_dim: int = 1792
    attention_dim: int = 512
    embed_dim: int = 512
    decoder_dim: int = 512
    dropout: float = 0.5
    fine_tune_encoder_from_epoch: int = 10

    # training
    batch_size: int = 32
    epochs: int = 50
    lr_encoder: float = 1e-4
    lr_decoder: float = 4e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    teacher_forcing_ratio: float = 1.0
    scheduled_sampling_decay: float = 0.98
    early_stopping_patience: int = 15
    amp: bool = True
    num_workers: int = 4
    checkpoint_every_n_batches: int = 2000
    log_every: int = 1000
    val_metric_batches: int = 50

    # inference
    beam_sizes: tuple = (3, 5, 7, 10)

    device: str = "cpu"

    def ensure_dirs(self) -> None:
        for d in [
            self.cache_dir,
            self.checkpoint_dir,
            self.output_dir,
            self.runs_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config() -> Config:
    import torch

    cfg = Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.ensure_dirs()
    return cfg
