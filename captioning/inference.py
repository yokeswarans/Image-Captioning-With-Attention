from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import eval_transform
from .vocab import Vocabulary


@torch.no_grad()
def beam_search_caption(
    encoder,
    decoder,
    vocab,
    image_tensor,
    device,
    encoder_dim,
    beam_size=5,
    max_len=50,
):
    encoder.eval()
    decoder.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    enc_out = encoder(image_tensor).view(1, -1, encoder_dim)
    num_pixels = enc_out.size(1)

    k = beam_size
    enc_out = enc_out.expand(k, num_pixels, encoder_dim)

    start_id = vocab.word2idx[Vocabulary.START]
    end_id = vocab.word2idx[Vocabulary.END]

    seqs = torch.full((k, 1), start_id, dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, device=device)
    seq_alphas = torch.zeros(k, 1, enc_out.size(1), device=device)

    h, c = decoder.init_hidden_state(enc_out)
    complete_seqs, complete_scores, complete_alphas = [], [], []

    step = 1
    k_prev_words = torch.full((k, 1), start_id, dtype=torch.long, device=device)

    while True:
        embed = decoder.embedding(k_prev_words).squeeze(1)
        context, alpha = decoder.attention(enc_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context
        h, c = decoder.lstm_cell(torch.cat([embed, context], dim=1), (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // len(vocab)
        next_word_inds = top_k_words % len(vocab)

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seq_alphas = torch.cat([seq_alphas[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete = [i for i, w in enumerate(next_word_inds) if w.item() != end_id]
        complete = [i for i, w in enumerate(next_word_inds) if w.item() == end_id]

        if complete:
            complete_seqs.extend(seqs[complete].tolist())
            complete_scores.extend(top_k_scores[complete].tolist())
            complete_alphas.extend(seq_alphas[complete].tolist())
        k -= len(complete)
        if k == 0 or step >= max_len:
            break

        seqs = seqs[incomplete]
        seq_alphas = seq_alphas[incomplete]
        h, c = h[prev_word_inds[incomplete]], c[prev_word_inds[incomplete]]
        enc_out = enc_out[prev_word_inds[incomplete]]
        top_k_scores = top_k_scores[incomplete]
        k_prev_words = next_word_inds[incomplete].unsqueeze(1)
        step += 1

    if not complete_scores:
        complete_seqs = seqs.tolist()
        complete_scores = top_k_scores.tolist()
        complete_alphas = seq_alphas.tolist()

    best_idx = int(np.argmax(complete_scores))
    best_seq = complete_seqs[best_idx][1:]
    best_alphas = complete_alphas[best_idx][1:]
    caption = vocab.decode(best_seq)
    return caption, best_seq, best_alphas


def visualize_attention(image: Image.Image, word_ids, alphas, vocab, image_path, beam_size, output_dir, grid=7):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom as ndi_zoom

    os.makedirs(output_dir, exist_ok=True)
    words, kept_alphas = [], []
    for i, a in zip(word_ids, alphas):
        w = vocab.idx2word.get(int(i), Vocabulary.UNK)
        if w == Vocabulary.END:
            break
        if w in (Vocabulary.START, Vocabulary.PAD):
            continue
        words.append(w)
        kept_alphas.append(a)
    alphas = kept_alphas
    image = image.resize((336, 336))
    n = len(words)
    cols = min(5, max(1, n))
    rows = math.ceil(n / cols) if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for t, word in enumerate(words):
        ax = axes[t]
        ax.imshow(image)
        alpha_map = np.array(alphas[t]).reshape(grid, grid)
        alpha_img = ndi_zoom(alpha_map, 336 / grid, order=3)
        ax.imshow(alpha_img, alpha=0.6, cmap="jet")
        ax.set_title(word, fontsize=10)
        ax.axis("off")
    for ax in axes[len(words) :]:
        ax.axis("off")

    plt.tight_layout()
    stem = Path(image_path).stem
    att_path = os.path.join(output_dir, f"{stem}_beam_k{beam_size}_attention.png")
    plt.savefig(att_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(image)
    ax2.axis("off")
    ax2.set_title(" ".join(words), fontsize=11, wrap=True)
    cap_path = os.path.join(output_dir, f"{stem}_beam_k{beam_size}_caption.png")
    plt.savefig(cap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    return cap_path, att_path


def predict_image(
    encoder,
    decoder,
    vocab,
    image_path,
    device,
    encoder_dim,
    max_caption_len,
    output_dir,
    beam_size=5,
    visualize=False,
):
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image)
    caption, ids, alphas = beam_search_caption(
        encoder,
        decoder,
        vocab,
        tensor,
        device,
        encoder_dim,
        beam_size=beam_size,
        max_len=max_caption_len,
    )

    sep = "─" * 55
    print(sep)
    print(f"  Image   : {Path(image_path).name}")
    print(f"  Mode    : beam_k{beam_size}")
    print(f"  Caption : {caption}")
    print(sep)

    cap_path = att_path = None
    if visualize:
        cap_path, att_path = visualize_attention(
            image, ids, alphas, vocab, image_path, beam_size, output_dir
        )
        print(f"Caption figure saved → {cap_path}")
        print(f"Attention map saved → {att_path}")
        print(f"Generated files are saved in {output_dir}/.")

    return caption, ids, alphas, cap_path, att_path
