#!/usr/bin/env python3
"""Phase 2 training entrypoint.

Usage:
  python train.py --patience 15 --epoch 80
  python train.py --patience 10 --epoch 80 --resume
  python train.py --patience 15 --epoch 80 --log_every 1000

Artifacts:
  runs/train.log          — full training log (batch + epoch)
  runs/STATUS             — RUNNING / COMPLETED / EARLY_STOPPED / FAILED
  runs/status.json        — same status as JSON
  runs/history.json       — per-epoch metrics
  runs/train_val_loss.png — loss curves (updated each epoch)
  runs/bleu_scores.png
  runs/other_metrics.png
  runs/schedule.png
  checkpoints/last_model.pth
  checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from captioning.checkpoint import load_checkpoint, save_checkpoint
from captioning.config import get_default_config
from captioning.dataset import build_dataloaders, exclude_training_images_from_validation
from captioning.metrics import compute_val_loss, evaluate
from captioning.model import build_models
from captioning.utils import HistoryTracker, setup_logger, write_status
from captioning.vocab import Vocabulary, build_or_load_vocab_and_captions

# Notebooks saved Vocabulary under __main__; alias so old .pth files unpickle.
sys.modules["__main__"].Vocabulary = Vocabulary


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    cfg = get_default_config()
    p = argparse.ArgumentParser(description="Train image captioning model (Phase 2)")
    p.add_argument("--epochs", "--epoch", type=int, default=cfg.epochs, dest="epochs",
                   help="Max training epochs (e.g. --epoch 80)")
    p.add_argument("--patience", type=int, default=cfg.early_stopping_patience, help="Early stopping patience")
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr_decoder", type=float, default=cfg.lr_decoder)
    p.add_argument("--lr_encoder", type=float, default=cfg.lr_encoder)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--log_every", type=int, default=cfg.log_every, help="Log every N batches (default: 1000)")
    p.add_argument("--val_batches", type=int, default=cfg.val_metric_batches, help="Max val batches for BLEU eval")
    p.add_argument("--checkpoint_every", type=int, default=cfg.checkpoint_every_n_batches)
    p.add_argument("--fine_tune_from", type=int, default=cfg.fine_tune_encoder_from_epoch)
    p.add_argument("--runs_dir", type=str, default=cfg.runs_dir)
    p.add_argument("--checkpoint_dir", type=str, default=cfg.checkpoint_dir)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/last_model.pth")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = get_default_config()
    cfg.early_stopping_patience = args.patience
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr_decoder = args.lr_decoder
    cfg.lr_encoder = args.lr_encoder
    cfg.num_workers = args.num_workers
    cfg.log_every = max(1, args.log_every)
    cfg.val_metric_batches = args.val_batches
    cfg.checkpoint_every_n_batches = args.checkpoint_every
    cfg.fine_tune_encoder_from_epoch = args.fine_tune_from
    cfg.runs_dir = args.runs_dir
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.amp = not args.no_amp
    if args.device:
        cfg.device = args.device
    cfg.ensure_dirs()

    device = torch.device(cfg.device)
    logger = setup_logger(cfg.runs_dir)
    history = HistoryTracker(cfg.runs_dir)

    write_status(cfg.runs_dir, "RUNNING", {"epoch": None, "message": "starting up"})
    logger.info("=" * 72)
    logger.info("Image Captioning — Phase 2 training started")
    logger.info("PID=%s  device=%s  patience=%s  epochs=%s  batch_size=%s  amp=%s",
                os.getpid(), device, cfg.early_stopping_patience, cfg.epochs, cfg.batch_size, cfg.amp)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("Config: %s", cfg.to_dict())
    logger.info("Monitor remotely:")
    logger.info("  tail -f %s", os.path.join(cfg.runs_dir, "train.log"))
    logger.info("  cat %s", os.path.join(cfg.runs_dir, "STATUS"))
    logger.info("=" * 72)

    try:
        train_id2clean, train_id2file, vocab = build_or_load_vocab_and_captions(
            cfg.train_ann, cfg.cache_dir, "train", cfg.min_word_freq, logger
        )
        val_id2clean, val_id2file = build_or_load_vocab_and_captions(
            cfg.val_ann, cfg.cache_dir, "val", cfg.min_word_freq, logger
        )
        val_id2clean, val_id2file, overlapping_val_ids = exclude_training_images_from_validation(
            train_id2file, val_id2clean, val_id2file
        )
        if overlapping_val_ids:
            logger.warning(
                "Excluded %d validation images also present in training annotations; sample IDs: %s",
                len(overlapping_val_ids),
                sorted(overlapping_val_ids)[:10],
            )
        else:
            logger.info("Training and validation image IDs are disjoint")
        logger.info("Vocab size: %d", len(vocab))

        train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
            cfg, train_id2clean, train_id2file, val_id2clean, val_id2file, vocab
        )
        logger.info("Train samples (image,caption pairs): %d", len(train_dataset))
        logger.info("Val samples: %d", len(val_dataset))
        logger.info("Batches per epoch: %d", len(train_loader))

        encoder, decoder = build_models(cfg, len(vocab), device, pretrained=True)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[Vocabulary.PAD])
        decoder_optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=cfg.lr_decoder, weight_decay=cfg.weight_decay
        )
        encoder_optimizer = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=cfg.epochs)
        scaler = GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

        start_epoch, best_bleu4, patience_counter = 0, 0.0, 0
        tf_ratio = cfg.teacher_forcing_ratio

        last_ckpt = os.path.join(cfg.checkpoint_dir, "last_model.pth")
        if args.resume and os.path.exists(last_ckpt):
            logger.info("Resuming from %s", last_ckpt)
            state = torch.load(last_ckpt, map_location=device, weights_only=False)
            if state.get("encoder_optimizer") is not None:
                encoder.fine_tune(True)
                encoder_optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, encoder.parameters()),
                    lr=cfg.lr_encoder,
                    weight_decay=cfg.weight_decay,
                )
            state = load_checkpoint(
                last_ckpt, encoder, decoder, decoder_optimizer, encoder_optimizer, map_location=device
            )
            start_epoch = int(state["epoch"]) + 1
            best_bleu4 = float(state.get("best_bleu4", 0.0))
            tf_ratio = float(state.get("tf_ratio", cfg.teacher_forcing_ratio))
            patience_counter = int(state.get("patience_counter", 0))
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(
                "Resumed at epoch %d | best BLEU-4=%.4f | patience_counter=%d | tf_ratio=%.3f",
                start_epoch,
                best_bleu4,
                patience_counter,
                tf_ratio,
            )
        elif args.resume:
            logger.warning("--resume requested but %s not found; starting fresh", last_ckpt)

        def current_state(epoch, best_bleu4_val, patience_ctr, tf):
            return {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict() if encoder_optimizer else None,
                "best_bleu4": best_bleu4_val,
                "patience_counter": patience_ctr,
                "tf_ratio": tf,
                "vocab": vocab,
                "config": cfg.to_dict(),
            }

        def train_one_epoch(epoch, teacher_forcing_ratio, best_bleu4_so_far, patience_ctr):
            encoder.train()
            decoder.train()
            running_loss, n_batches = 0.0, 0
            t_epoch = time.time()
            num_batches = len(train_loader)

            for i, (images, captions, lengths, _) in enumerate(train_loader):
                t0 = time.time()
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                decoder_optimizer.zero_grad(set_to_none=True)
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad(set_to_none=True)

                with autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                    encoder_out = encoder(images)
                    predictions, alphas, decode_lengths = decoder(
                        encoder_out, captions, lengths, teacher_forcing_ratio
                    )
                    targets = captions[:, 1:]
                    packed_preds = torch.cat(
                        [predictions[b, : decode_lengths[b], :] for b in range(len(decode_lengths))],
                        dim=0,
                    )
                    packed_targets = torch.cat(
                        [targets[b, : decode_lengths[b]] for b in range(len(decode_lengths))],
                        dim=0,
                    )
                    loss = criterion(packed_preds, packed_targets)
                    loss = loss + 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

                scaler.scale(loss).backward()
                scaler.unscale_(decoder_optimizer)
                nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
                if encoder_optimizer is not None:
                    scaler.unscale_(encoder_optimizer)
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, encoder.parameters()), cfg.grad_clip
                    )

                scaler.step(decoder_optimizer)
                if encoder_optimizer is not None:
                    scaler.step(encoder_optimizer)
                scaler.update()

                running_loss += loss.item()
                n_batches += 1
                batch_time = time.time() - t0

                if (i + 1) % cfg.log_every == 0 or i + 1 == num_batches:
                    avg = running_loss / n_batches
                    eta_s = batch_time * (num_batches - i - 1)
                    logger.info(
                        "EPOCH %d | BATCH %d/%d | loss=%.4f | avg_loss=%.4f | batch_time=%.2fs | ETA_epoch=%.0fs",
                        epoch,
                        i + 1,
                        num_batches,
                        loss.item(),
                        avg,
                        batch_time,
                        eta_s,
                    )
                    write_status(
                        cfg.runs_dir,
                        "RUNNING",
                        {
                            "epoch": epoch,
                            "batch": i + 1,
                            "num_batches": num_batches,
                            "batch_loss": round(loss.item(), 4),
                            "avg_train_loss": round(avg, 4),
                            "best_bleu4": round(best_bleu4_so_far, 4),
                        },
                    )

                if cfg.checkpoint_every_n_batches and i > 0 and i % cfg.checkpoint_every_n_batches == 0:
                    path, _ = save_checkpoint(
                        current_state(epoch, best_bleu4_so_far, patience_ctr, teacher_forcing_ratio),
                        is_best=False,
                        checkpoint_dir=cfg.checkpoint_dir,
                    )
                    logger.info("Mid-epoch safety checkpoint saved → %s (batch %d)", path, i)

            return running_loss / max(n_batches, 1), time.time() - t_epoch

        final_status = "COMPLETED"
        for epoch in range(start_epoch, cfg.epochs):
            if epoch == cfg.fine_tune_encoder_from_epoch and encoder_optimizer is None:
                logger.info(">> Unfreezing encoder for fine-tuning (epoch %d)", epoch)
                encoder.fine_tune(True)
                encoder_optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, encoder.parameters()),
                    lr=cfg.lr_encoder,
                    weight_decay=cfg.weight_decay,
                )

            write_status(cfg.runs_dir, "RUNNING", {"epoch": epoch, "phase": "train"})
            logger.info("-" * 72)
            logger.info("Starting epoch %d / %d  (tf_ratio=%.3f)", epoch, cfg.epochs - 1, tf_ratio)

            train_loss, train_time = train_one_epoch(epoch, tf_ratio, best_bleu4, patience_counter)
            logger.info(
                "Epoch %d train done | train_loss=%.4f | time=%.1fs",
                epoch,
                train_loss,
                train_time,
            )

            write_status(cfg.runs_dir, "RUNNING", {"epoch": epoch, "phase": "val_loss"})
            logger.info("Computing validation loss ...")
            val_loss = compute_val_loss(
                encoder,
                decoder,
                val_loader,
                criterion,
                device,
                amp=cfg.amp,
                max_batches=cfg.val_metric_batches,
            )
            logger.info("val_loss=%.4f", val_loss)

            write_status(cfg.runs_dir, "RUNNING", {"epoch": epoch, "phase": "val_metrics"})
            logger.info("Evaluating metrics (max_batches=%d) ...", cfg.val_metric_batches)
            val_metrics = evaluate(
                encoder, decoder, val_loader, vocab, cfg, device, max_batches=cfg.val_metric_batches
            )
            for k, v in val_metrics.items():
                logger.info("  %s = %.4f", k, v)

            lr_now = decoder_optimizer.param_groups[0]["lr"]
            scheduler.step()
            tf_ratio *= cfg.scheduled_sampling_decay

            bleu4 = val_metrics["BLEU-4"]
            is_best = bleu4 > best_bleu4
            if is_best:
                best_bleu4 = bleu4
                patience_counter = 0
                logger.info("★ New best BLEU-4: %.4f", best_bleu4)
            else:
                patience_counter += 1
                logger.info(
                    "No BLEU-4 improvement (%d / %d patience)",
                    patience_counter,
                    cfg.early_stopping_patience,
                )

            last_path, best_path = save_checkpoint(
                current_state(epoch, best_bleu4, patience_counter, tf_ratio),
                is_best=is_best,
                checkpoint_dir=cfg.checkpoint_dir,
            )
            logger.info("Saved last checkpoint → %s", last_path)
            if best_path:
                logger.info("Saved best checkpoint → %s", best_path)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tf_ratio": tf_ratio,
                "lr_decoder": lr_now,
                "best_bleu4": best_bleu4,
                "patience_counter": patience_counter,
                "train_time_sec": round(train_time, 1),
                **val_metrics,
            }
            history.append(record)
            logger.info(
                "EPOCH COMPLETE %d | train_loss=%.4f val_loss=%.4f BLEU-4=%.4f (best=%.4f) | graphs updated in %s/",
                epoch,
                train_loss,
                val_loss,
                bleu4,
                best_bleu4,
                cfg.runs_dir,
            )

            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    cfg.early_stopping_patience,
                )
                final_status = "EARLY_STOPPED"
                break

        write_status(
            cfg.runs_dir,
            final_status,
            {"best_bleu4": round(best_bleu4, 4), "last_epoch": history.records[-1]["epoch"] if history.records else None},
        )
        logger.info("=" * 72)
        logger.info("Training finished with status=%s | best_bleu4=%.4f", final_status, best_bleu4)
        logger.info("Checkpoints: %s/{last_model.pth,best_model.pth}", cfg.checkpoint_dir)
        logger.info("Graphs/logs: %s/", cfg.runs_dir)
        logger.info("=" * 72)
        return 0

    except KeyboardInterrupt:
        write_status(cfg.runs_dir, "STOPPED", {"message": "KeyboardInterrupt"})
        logger.warning("Training interrupted by user (KeyboardInterrupt)")
        return 130
    except Exception as e:
        write_status(cfg.runs_dir, "FAILED", {"error": str(e)})
        logger.error("Training FAILED: %s", e)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
