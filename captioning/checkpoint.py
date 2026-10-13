from __future__ import annotations

import os

import torch


def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        return last_path, best_path
    return last_path, None


def load_checkpoint(path, encoder, decoder, decoder_optimizer=None, encoder_optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    if decoder_optimizer is not None and state.get("decoder_optimizer") is not None:
        decoder_optimizer.load_state_dict(state["decoder_optimizer"])
    if encoder_optimizer is not None and state.get("encoder_optimizer") is not None:
        encoder_optimizer.load_state_dict(state["encoder_optimizer"])
    return state
