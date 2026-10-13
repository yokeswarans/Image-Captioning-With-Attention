#!/usr/bin/env python3
"""Phase 2 prediction / attention visualisation entrypoint.

Usage:
  python predict.py --image photo.jpg
  python predict.py --image photo.jpg --beam_size 7
  python predict.py --image photo.jpg --visualize
  python predict.py --image photo.jpg --beam_size 5 --visualize --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from captioning.checkpoint import load_checkpoint
from captioning.config import get_default_config
from captioning.inference import predict_image
from captioning.model import build_models
from captioning.vocab import Vocabulary

# Notebooks saved Vocabulary under __main__; alias so old .pth files unpickle.
sys.modules["__main__"].Vocabulary = Vocabulary


def parse_args():
    cfg = get_default_config()
    p = argparse.ArgumentParser(description="Generate captions with beam search + optional attention maps")
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--beam_size", type=int, default=5, help="Beam size (e.g. 3, 5, 7, 10)")
    p.add_argument("--visualize", action="store_true", help="Save attention heatmaps + caption figure")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(cfg.checkpoint_dir, "best_model.pth"),
        help="Model checkpoint (default: checkpoints/best_model.pth)",
    )
    p.add_argument("--output_dir", type=str, default=cfg.output_dir)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_default_config()
    if args.device:
        cfg.device = args.device
    cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    device = torch.device(cfg.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = state["vocab"]
    saved_cfg = state.get("config") or {}
    encoder_dim = int(saved_cfg.get("encoder_dim", cfg.encoder_dim))
    max_caption_len = int(saved_cfg.get("max_caption_len", cfg.max_caption_len))
    for k in ("attention_dim", "embed_dim", "decoder_dim", "encoder_dim", "dropout"):
        if k in saved_cfg:
            setattr(cfg, k, saved_cfg[k])

    encoder, decoder = build_models(cfg, len(vocab), device, pretrained=False)
    load_checkpoint(args.checkpoint, encoder, decoder, map_location=device)
    encoder.eval()
    decoder.eval()

    predict_image(
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        image_path=args.image,
        device=device,
        encoder_dim=encoder_dim,
        max_caption_len=max_caption_len,
        output_dir=cfg.output_dir,
        beam_size=args.beam_size,
        visualize=args.visualize,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
