#!/usr/bin/env python3
"""Interactive UI for the image-captioning project.

This file is additive only — it imports the existing `captioning` package
(model, checkpoint, vocab, transforms, beam search) and never modifies it.
`python predict.py --image ...` keeps working exactly as before, and still
writes its PNGs to outputs/predictions/ like it always has.

The UI itself never writes files to disk for captions/images: captions are
generated with beam_search_caption() straight from the in-memory
uploaded/fetched image, and the attention heatmap (when that toggle is on)
is rendered to an in-memory PNG buffer and handed directly to the browser.

The one deliberate exception is the attention *video* toggle — a browser
<video> player has to stream from a real file, so that one writes an .mp4
into outputs/predictions/videos/. To keep that bounded, only the 5 most
recent videos are ever kept on disk; every new save prunes the oldest ones.

Usage:
  python app.py
  python app.py --checkpoint checkpoints/last_model.pth
  python app.py --host 0.0.0.0 --port 7860 --share

Requires the `gradio` package — add it to requirements.txt:
  gradio>=4.0
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make sure the project root is importable no matter where this is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from captioning.checkpoint import load_checkpoint
from captioning.config import get_default_config
from captioning.dataset import eval_transform
from captioning.inference import beam_search_caption
from captioning.model import build_models
from captioning.vocab import Vocabulary

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "gradio is not installed. Add 'gradio>=4.0' to requirements.txt and "
        "run: pip install gradio"
    ) from e

# Same alias predict.py uses so old .pth files (pickled with Vocabulary under
# __main__ from the notebooks) unpickle correctly.
sys.modules["__main__"].Vocabulary = Vocabulary

APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "webui" / "theme.css"
BASE_CFG = get_default_config()

CHECKPOINT_CHOICES = [
    os.path.join(BASE_CFG.checkpoint_dir, "best_model.pth"),
    os.path.join(BASE_CFG.checkpoint_dir, "last_model.pth"),
]

EXAMPLE_IMAGES = [
    p for p in ["test.webp", "test1.webp", "download.webp", "monkry.webp"]
    if (APP_DIR / p).is_file()
]

# ----------------------------------------------------------------------------
# Model loading (lazy — first request only, so the UI still opens even if a
# checkpoint hasn't been produced yet).
# ----------------------------------------------------------------------------

_STATE = {
    "encoder": None,
    "decoder": None,
    "vocab": None,
    "cfg": None,
    "checkpoint": None,
    "device": None,
    "history": None,
    "vocab_words_sorted": None,
    "embed_matrix": None,
    "embed_word_to_row": None,
    "embed_words": None,
}


def _pick_checkpoint(explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit
    for cand in CHECKPOINT_CHOICES:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "No checkpoint found. Expected one of: "
        + ", ".join(CHECKPOINT_CHOICES)
        + ". Train the model (train.py) or pass --checkpoint."
    )


def load_model(explicit_checkpoint: str | None = None, device_override: str | None = None) -> None:
    if _STATE["encoder"] is not None:
        return

    ckpt_path = _pick_checkpoint(explicit_checkpoint)
    device = torch.device(device_override or BASE_CFG.device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab = state["vocab"]
    saved_cfg = state.get("config") or {}
    cfg = get_default_config()
    for k in ("attention_dim", "embed_dim", "decoder_dim", "encoder_dim", "dropout", "max_caption_len"):
        if k in saved_cfg:
            setattr(cfg, k, saved_cfg[k])

    encoder, decoder = build_models(cfg, len(vocab), device, pretrained=False)
    load_checkpoint(ckpt_path, encoder, decoder, map_location=device)
    encoder.eval()
    decoder.eval()

    _STATE.update(
        encoder=encoder,
        decoder=decoder,
        vocab=vocab,
        cfg=cfg,
        checkpoint=ckpt_path,
        device=device,
    )


def _status_line() -> str:
    try:
        load_model()
        name = os.path.basename(_STATE["checkpoint"])
        vocab_size = len(_STATE["vocab"])
        return (
            f'<span class="status-line">● model ready &nbsp;·&nbsp; '
            f"<b>{name}</b> &nbsp;·&nbsp; device <b>{_STATE['device'].type}</b> "
            f"&nbsp;·&nbsp; vocab <b>{vocab_size:,}</b> tokens</span>"
        )
    except Exception as e:
        return f'<span class="status-line" style="color:#ff8383">○ no checkpoint loaded — {e}</span>'


def _pipeline_html() -> str:
    """A visual outline of the actual architecture, built from config.py — not
    hardcoded copy, so it stays true if the config ever changes."""
    return f"""
<div class="pipeline-strip">
  <div class="pipeline-chip">
    <span class="chip-role">Encoder</span>
    <span class="chip-name">EfficientNet&#8209;B4</span>
    <span class="chip-spec">{BASE_CFG.encoder_dim}-d feature grid &middot; 7&times;7 patches</span>
  </div>
  <div class="pipeline-arrow">&#8594;</div>
  <div class="pipeline-chip">
    <span class="chip-role">Attention</span>
    <span class="chip-name">Bahdanau (additive)</span>
    <span class="chip-spec">{BASE_CFG.attention_dim}-d alignment &middot; soft attention</span>
  </div>
  <div class="pipeline-arrow">&#8594;</div>
  <div class="pipeline-chip">
    <span class="chip-role">Decoder</span>
    <span class="chip-name">LSTM cell</span>
    <span class="chip-spec">{BASE_CFG.decoder_dim}-d hidden &middot; beam search</span>
  </div>
</div>
"""


def _footer_readout() -> str:
    """Pull best-epoch metrics from runs/history.json for a small readout, if present."""
    history_path = APP_DIR / BASE_CFG.runs_dir / "history.json"
    if not history_path.is_file():
        return ""
    try:
        import json

        records = json.loads(history_path.read_text())
        if not records:
            return ""
        best = max(records, key=lambda r: r.get("BLEU-4", 0.0))
        last = records[-1]
        return (
            "**Training readout**\n\n"
            f"- best epoch: `{best['epoch']}`  ·  BLEU-4 `{best.get('BLEU-4', 0):.3f}`  ·  "
            f"CIDEr-D `{best.get('CIDEr-D', 0):.3f}`  ·  METEOR `{best.get('METEOR', 0):.3f}`\n"
            f"- last recorded epoch: `{last['epoch']}`  ·  train_loss `{last.get('train_loss', 0):.3f}`  ·  "
            f"val_loss `{last.get('val_loss', 0):.3f}`"
        )
    except Exception:
        return ""


MAX_URL_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB safety cap
URL_FETCH_TIMEOUT = 12  # seconds


# ----------------------------------------------------------------------------
# Vocabulary / dictionary search
# ----------------------------------------------------------------------------
#
# The trained Vocabulary object (word2idx / idx2word) is already sitting in
# memory as soon as the checkpoint loads — this just exposes it as a
# prefix-searchable list. No DB, no extra file, nothing new to load.

DICT_MAX_RESULTS = 150


def _sorted_vocab_words() -> list[str]:
    if _STATE["vocab_words_sorted"] is None:
        vocab = _STATE["vocab"]
        specials = {Vocabulary.PAD, Vocabulary.START, Vocabulary.END, Vocabulary.UNK}
        _STATE["vocab_words_sorted"] = sorted(w for w in vocab.word2idx if w not in specials)
    return _STATE["vocab_words_sorted"]


def search_vocab(prefix: str) -> str:
    prefix = (prefix or "").strip().lower()

    try:
        load_model()
    except Exception as e:
        return f"*Dictionary unavailable — {e}*"

    words = _sorted_vocab_words()

    if not prefix:
        return f"*Type a letter or a few — e.g. `y`, then `ye` — to search the **{len(words):,}**-word vocabulary this model was trained on.*"

    matches = [w for w in words if w.startswith(prefix)]

    if not matches:
        return (
            f'**"{prefix}"** — no match.\n\n'
            f"Nothing starting with that is in the trained vocabulary — the model has "
            f"never seen it and cannot generate it. Expect captions to fall back to the "
            f"closest concept it *does* know instead."
        )

    shown = matches[:DICT_MAX_RESULTS]
    chips = " ".join(f"`{w}`" for w in shown)
    tail = f"\n\n*(+{len(matches) - DICT_MAX_RESULTS:,} more not shown)*" if len(matches) > DICT_MAX_RESULTS else ""
    plural = "word" if len(matches) == 1 else "words"
    return f'**{len(matches):,} {plural}** starting with **"{prefix}"**\n\n{chips}{tail}'


def toggle_dictionary(is_open: bool):
    new_state = not is_open
    label = "📖 Hide dictionary" if new_state else "📖 Dictionary"
    return new_state, gr.Group(visible=new_state), gr.Button(value=label)


# ----------------------------------------------------------------------------
# Embedding space — cosine similarity, nearest neighbors, and a local PCA
# scatter. All of this reads decoder.embedding.weight — the model's already-
# trained word-embedding matrix, sitting in memory the moment the checkpoint
# loads. No new inference, no disk writes. Note: this is the DECODER's input
# embedding space, learned purely from next-word prediction on COCO captions
# — a captioning-specific space, not a general semantic one like GloVe/word2vec.
# ----------------------------------------------------------------------------

EMBED_NEIGHBORS_K = 6
EMBED_PLOT_SIZE = 380


def _embedding_lookup() -> tuple[np.ndarray, dict]:
    """Returns (unit-normalized embedding matrix, word->row index), cached.
    Excludes the special tokens (<pad>/<start>/<end>/<unk>) — comparing
    against those isn't meaningful."""
    if _STATE.get("embed_matrix") is not None:
        return _STATE["embed_matrix"], _STATE["embed_word_to_row"]

    vocab = _STATE["vocab"]
    weight = _STATE["decoder"].embedding.weight.detach().cpu().numpy()
    specials = {Vocabulary.PAD, Vocabulary.START, Vocabulary.END, Vocabulary.UNK}
    words = [w for w in _sorted_vocab_words() if w not in specials]

    rows = np.stack([weight[vocab.word2idx[w]] for w in words])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = rows / norms
    word_to_row = {w: i for i, w in enumerate(words)}

    _STATE["embed_matrix"] = unit
    _STATE["embed_word_to_row"] = word_to_row
    _STATE["embed_words"] = words
    return unit, word_to_row


def _nearest_neighbors(word: str, k: int = EMBED_NEIGHBORS_K) -> list[tuple[str, float]]:
    unit, word_to_row = _embedding_lookup()
    words = _STATE["embed_words"]
    idx = word_to_row[word]
    sims = unit @ unit[idx]
    order = np.argsort(-sims)
    out = []
    for i in order:
        if words[i] == word:
            continue
        out.append((words[i], float(sims[i])))
        if len(out) >= k:
            break
    return out


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Plain numpy PCA (via SVD) — a genuine linear projection, no extra
    dependency. Distances in the result are proportionally faithful to the
    real embedding space, unlike t-SNE, which can visually distort them."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def render_embedding_plot(word_a: str, word_b: str, neighbors_a, neighbors_b) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit, word_to_row = _embedding_lookup()
    neighbor_words = [w for w, _ in neighbors_a] + [w for w, _ in neighbors_b]
    local_words = [word_a, word_b] + [w for w in neighbor_words if w not in (word_a, word_b)]
    seen = set()
    local_words = [w for w in local_words if not (w in seen or seen.add(w))]

    vectors = np.stack([unit[word_to_row[w]] for w in local_words])
    coords = _pca_2d(vectors)

    fig, ax = plt.subplots(figsize=(EMBED_PLOT_SIZE / 100, EMBED_PLOT_SIZE / 100), dpi=100)
    fig.patch.set_facecolor("#080b14")
    ax.set_facecolor("#080b14")

    for i, w in enumerate(local_words):
        x, y = coords[i]
        if w == word_a:
            ax.scatter(x, y, s=90, color="#5eead4", zorder=3)
            ax.text(x, y + 0.02, w, color="#5eead4", fontsize=12, fontweight="bold", ha="center")
        elif w == word_b:
            ax.scatter(x, y, s=90, color="#ffb454", zorder=3)
            ax.text(x, y + 0.02, w, color="#ffb454", fontsize=12, fontweight="bold", ha="center")
        else:
            ax.scatter(x, y, s=28, color="#5c6789", zorder=2)
            ax.text(x, y + 0.015, w, color="#96a2c2", fontsize=9, ha="center")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#232c44")
    fig.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="#080b14")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compare_words(word_a: str, word_b: str):
    word_a = (word_a or "").strip().lower()
    word_b = (word_b or "").strip().lower()

    if not word_a or not word_b:
        return "*Enter two words above, then press Compare.*", gr.Image(value=None, visible=False)

    try:
        load_model()
    except Exception as e:
        return f"*Embedding space unavailable — {e}*", gr.Image(value=None, visible=False)

    vocab = _STATE["vocab"]
    not_found = [w for w in (word_a, word_b) if w not in vocab.word2idx]
    if not_found:
        listed = ", ".join(f'"{w}"' for w in not_found)
        return (
            f"**{listed}** — not in the trained vocabulary, so it has no embedding to compare. "
            f"Try the dictionary search above to find words that *are* in it.",
            gr.Image(value=None, visible=False),
        )

    unit, word_to_row = _embedding_lookup()
    vec_a, vec_b = unit[word_to_row[word_a]], unit[word_to_row[word_b]]
    cosine = float(np.dot(vec_a, vec_b))

    neighbors_a = _nearest_neighbors(word_a)
    neighbors_b = _nearest_neighbors(word_b)
    chips_a = " ".join(f"`{w}`" for w, _ in neighbors_a)
    chips_b = " ".join(f"`{w}`" for w, _ in neighbors_b)

    summary = (
        f"**{cosine:.3f}** cosine similarity between **{word_a}** and **{word_b}**\n\n"
        f"closest to **{word_a}**: {chips_a}\n\n"
        f"closest to **{word_b}**: {chips_b}"
    )

    plot_img = render_embedding_plot(word_a, word_b, neighbors_a, neighbors_b)
    return summary, gr.Image(value=plot_img, visible=True)


def fetch_image_from_url(url: str) -> Image.Image:
    """Download an image from an arbitrary http(s) URL (e.g. a Google Images
    result link) and return it as a PIL image. Stdlib-only — no new
    dependency needed."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Enter an image URL first.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    # A browser-like User-Agent avoids 403s from sites that block bare urllib requests.
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=URL_FETCH_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if content_type and "image" not in content_type:
                raise ValueError(f"That URL didn't return an image (got '{content_type}').")
            data = resp.read(MAX_URL_IMAGE_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise ValueError(f"Server returned {e.code} for that URL.") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Couldn't reach that URL ({e.reason}).") from e

    if len(data) > MAX_URL_IMAGE_BYTES:
        raise ValueError("Image is larger than the 25 MB limit.")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError("Couldn't decode that URL as an image.") from e


def load_url_preview(url: str):
    try:
        image = fetch_image_from_url(url)
    except Exception as e:
        raise gr.Error(str(e))
    return gr.Image(value=image, visible=True)


def toggle_source_mode(mode: str):
    is_upload = mode == "Upload"
    return (
        gr.Group(visible=is_upload),
        gr.Group(visible=not is_upload),
    )


# ----------------------------------------------------------------------------
# In-memory attention rendering
# ----------------------------------------------------------------------------
#
# captioning/inference.py's visualize_attention() always writes its figure to
# a .png on disk (that's what predict.py needs). For the UI we want the exact
# same heatmap panel but with nothing touching disk, so this renders it to an
# in-memory buffer instead. Logic mirrors visualize_attention() 1:1; nothing
# in captioning/ is changed.

def _filtered_words_and_alphas(word_ids, alphas, vocab):
    """Strip <start>/<pad>/<end> the same way visualize_attention() does."""
    words, kept_alphas = [], []
    for i, a in zip(word_ids, alphas):
        w = vocab.idx2word.get(int(i), Vocabulary.UNK)
        if w == Vocabulary.END:
            break
        if w in (Vocabulary.START, Vocabulary.PAD):
            continue
        words.append(w)
        kept_alphas.append(a)
    return words, kept_alphas


def render_attention_inmemory(image: Image.Image, word_ids, alphas, vocab, grid: int = 7) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom as ndi_zoom

    words, kept_alphas = _filtered_words_and_alphas(word_ids, alphas, vocab)

    resized = image.resize((336, 336))
    n = len(words)
    cols = min(5, max(1, n))
    rows = math.ceil(n / cols) if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for t, word in enumerate(words):
        ax = axes[t]
        ax.imshow(resized)
        alpha_map = np.array(kept_alphas[t]).reshape(grid, grid)
        alpha_img = ndi_zoom(alpha_map, 336 / grid, order=3)
        ax.imshow(alpha_img, alpha=0.6, cmap="jet")
        ax.set_title(word, fontsize=10)
        ax.axis("off")
    for ax in axes[len(words):]:
        ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------------
# Attention video — the one deliberate exception to "nothing written to disk".
# Browsers can't stream video from an in-memory buffer the way gr.Image takes
# a PIL object, so this writes a real .mp4. To keep disk usage bounded (this
# matters a lot on a Hugging Face Space with limited/ephemeral storage), only
# the VIDEO_MAX_FILES most recent videos are ever kept — every new save
# prunes the oldest ones first.
# ----------------------------------------------------------------------------

VIDEO_DIR = Path(BASE_CFG.output_dir) / "videos"
VIDEO_MAX_FILES = 5
VIDEO_FPS = 1.0  # baseline encode rate; the UI's speed pill scales playback client-side on top of this


def _cleanup_old_videos(keep: int = VIDEO_MAX_FILES) -> None:
    if not VIDEO_DIR.is_dir():
        return
    files = sorted(VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    excess = len(files) - keep
    for f in files[: max(0, excess)]:
        try:
            f.unlink()
        except OSError:
            pass


def render_attention_video(image: Image.Image, word_ids, alphas, vocab, beam_size: int, grid: int = 7) -> str:
    try:
        import imageio
    except ImportError as e:
        raise RuntimeError(
            "The 'imageio' and 'imageio-ffmpeg' packages are required for attention video. "
            "Add them to requirements.txt and pip install them."
        ) from e

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom as ndi_zoom

    words, kept_alphas = _filtered_words_and_alphas(word_ids, alphas, vocab)
    if not words:
        raise ValueError("No words to visualize.")

    total = len(words)
    size = 420
    resized = image.resize((size, size))

    frames = []
    for idx, (word, a) in enumerate(zip(words, kept_alphas), start=1):
        fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # axes fills the whole canvas — no margins to clip anything into
        ax.imshow(resized)
        alpha_map = np.array(a).reshape(grid, grid)
        alpha_img = ndi_zoom(alpha_map, size / grid, order=3)
        ax.imshow(alpha_img, alpha=0.6, cmap="jet")
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.axis("off")

        # Word label is drawn *inside* the axes (transAxes coords), not as a
        # plt title floating in the margin above it — a title's margin can
        # clip a word depending on font metrics per-frame; text inside the
        # axes bounds physically cannot be cropped out of the saved frame.
        label = f"{idx}/{total}  {word}"
        fontsize = 15 if len(label) <= 16 else 11
        ax.text(
            0.5, 0.045, label,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=dict(facecolor="black", alpha=0.72, pad=6, edgecolor="none"),
            clip_on=False,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frame = np.array(Image.open(buf).convert("RGB"))
        frames.append(frame)

    # Guard against any stray pixel-size drift between frames (encoder needs
    # every frame to share identical dimensions).
    target_h, target_w = frames[0].shape[:2]
    frames = [
        f if f.shape[:2] == (target_h, target_w)
        else np.array(Image.fromarray(f).resize((target_w, target_h)))
        for f in frames
    ]

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = VIDEO_DIR / f"attn_{timestamp}_beam{beam_size}.mp4"

    imageio.mimsave(str(out_path), frames, fps=VIDEO_FPS, macro_block_size=1)

    _cleanup_old_videos()
    return str(out_path)


# ----------------------------------------------------------------------------
# Inference callback
# ----------------------------------------------------------------------------

def generate_caption(
    mode: str,
    uploaded_image,
    url_text: str,
    url_preview_image,
    beam_choice: str,
    attention_choice: str,
    video_choice: str,
):
    if mode == "URL":
        image = url_preview_image
        if image is None:
            if not (url_text or "").strip():
                raise gr.Error("Enter an image URL first.")
            try:
                image = fetch_image_from_url(url_text)
            except Exception as e:
                raise gr.Error(str(e))
    else:
        image = uploaded_image

    if image is None:
        raise gr.Error("Upload an image, or switch to URL and paste an image link.")

    try:
        load_model()
    except Exception as e:
        raise gr.Error(str(e))

    cfg = _STATE["cfg"]
    beam_size = int(beam_choice)
    visualize = attention_choice == "On"
    want_video = video_choice == "On"

    # Nothing is written to disk here — not to outputs/predictions/, not even
    # to a temp folder. The tensor lives in memory and beam_search_caption()
    # (unmodified, from captioning/inference.py) returns everything we need.
    tensor = eval_transform(image.convert("RGB"))

    t0 = time.time()
    caption, ids, alphas = beam_search_caption(
        _STATE["encoder"],
        _STATE["decoder"],
        _STATE["vocab"],
        tensor,
        _STATE["device"],
        cfg.encoder_dim,
        beam_size=beam_size,
        max_len=cfg.max_caption_len,
    )
    elapsed = time.time() - t0

    # Printed to the terminal running `python app.py`, same spirit as predict.py.
    print("─" * 55)
    print(f"  Source  : {mode}")
    print(f"  Mode    : beam_k{beam_size}")
    print(f"  Caption : {caption}")
    print("─" * 55)

    caption_md = f"{caption}"
    meta_md = (
        f"beam width {beam_size}  ·  {len(ids)} tokens  ·  "
        f"{elapsed:.2f}s  ·  {_STATE['device'].type}"
    )

    attention_update = gr.Image(value=None, visible=False)
    if visualize:
        attention_img = render_attention_inmemory(image, ids, alphas, _STATE["vocab"])
        attention_update = gr.Image(value=attention_img, visible=True)

    video_update = gr.Video(value=None, visible=False)
    if want_video:
        try:
            video_path = render_attention_video(image, ids, alphas, _STATE["vocab"], beam_size)
            video_update = gr.Video(value=video_path, visible=True)
        except Exception as e:
            raise gr.Error(f"Couldn't build attention video: {e}")

    return (
        gr.Markdown(value=caption_md, elem_classes=["caption-output"]),
        gr.Markdown(value=meta_md, elem_classes=["meta-line"]),
        attention_update,
        video_update,
    )


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

ADD_SCANNING_JS = "() => { document.querySelector('.app-shell')?.classList.add('is-generating'); }"
REMOVE_SCANNING_JS = "() => { document.querySelector('.app-shell')?.classList.remove('is-generating'); }"

# video.playbackRate is a browser-native property — switching speed never
# re-encodes or re-fetches anything, it just changes how fast the *same*
# already-loaded video plays.
APPLY_SPEED_JS = """(speed) => {
  const setRate = () => {
    const wrap = document.getElementById('attn-video');
    const v = wrap ? wrap.querySelector('video') : null;
    if (v) { v.playbackRate = parseFloat(speed) || 1; }
  };
  setRate();
  setTimeout(setRate, 300);
  return [];
}"""

HERO_HTML = """
<div class="hero">
  <div class="eyebrow"><span class="dot"></span>Image Captioning Project</div>
  <h1>EfficientNet&#8209;B4 &nbsp;+&nbsp; Bahdanau Attention &nbsp;+&nbsp; LSTM Decoder</h1>
  <p>Upload a photo or paste an image URL, generate a caption with beam search, and — if you want it —
  watch attention sweep across the same 7&times;7 grid the encoder actually reasons over.</p>
</div>
"""


def build_app() -> "gr.Blocks":
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.is_file() else ""

    with gr.Blocks(css=css, title="Image Captioning — Attention Field", theme=gr.themes.Base()) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(HERO_HTML)
            gr.HTML(_pipeline_html())
            status_html = gr.HTML(_status_line())

            dict_state = gr.State(False)
            with gr.Row(elem_classes=["dict-toggle-row"]):
                dict_toggle_btn = gr.Button("📖 Dictionary", elem_classes=["btn-secondary", "dict-toggle"])

            with gr.Group(visible=False, elem_classes=["glass-card", "dict-panel"]) as dict_panel:
                gr.Markdown(
                    '<div class="field-label">Vocabulary search '
                    '<span class="hint">prefix match over the trained ~10k-word vocabulary</span></div>'
                )
                dict_search = gr.Textbox(
                    placeholder='Type a prefix, e.g. "y" then "ye"…',
                    show_label=False,
                    elem_classes=["url-box"],
                )
                dict_results = gr.Markdown("", elem_classes=["dict-results"])

                gr.Markdown(
                    '<div class="field-label" style="margin-top:26px;">Embedding space '
                    '<span class="hint">compare two words from the search above</span></div>'
                )
                with gr.Row(elem_classes=["embed-input-row"]):
                    word_a_box = gr.Textbox(placeholder="word A, e.g. bus", show_label=False, elem_classes=["url-box"])
                    word_b_box = gr.Textbox(placeholder="word B, e.g. car", show_label=False, elem_classes=["url-box"])
                compare_btn = gr.Button("Compare in embedding space", elem_classes=["btn-secondary"])
                embed_results = gr.Markdown("", elem_classes=["dict-results"])
                embed_plot = gr.Image(
                    label="Local neighborhood (PCA projection)",
                    visible=False,
                    elem_classes=["attention-panel"],
                )

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["glass-card", "panel-left"]):
                    gr.Markdown(
                        '<div class="field-label">Image source <span class="hint">upload a file or paste a link</span></div>'
                    )
                    source_radio = gr.Radio(
                        choices=["Upload", "URL"],
                        value="Upload",
                        show_label=False,
                        elem_classes=["pill-radio"],
                    )

                    with gr.Group(visible=True) as upload_group:
                        image_in = gr.Image(
                            type="pil",
                            label=None,
                            show_label=False,
                            height=320,
                            elem_classes=["upload-zone"],
                        )

                    with gr.Group(visible=False) as url_group:
                        url_box = gr.Textbox(
                            placeholder="https://example.com/photo.jpg",
                            show_label=False,
                            elem_classes=["url-box"],
                        )
                        url_load_btn = gr.Button("Load preview", size="sm", elem_classes=["btn-secondary"])
                        url_preview = gr.Image(
                            type="pil",
                            label=None,
                            show_label=False,
                            visible=False,
                            height=280,
                            elem_classes=["upload-zone"],
                        )

                    gr.Markdown(
                        '<div class="field-label">Beam width <span class="hint">wider = more exploration</span></div>'
                    )
                    beam_radio = gr.Radio(
                        choices=["5", "7"],
                        value="5",
                        show_label=False,
                        elem_classes=["pill-radio"],
                    )

                    gr.Markdown(
                        '<div class="field-label">Attention view <span class="hint">overlay per word</span></div>'
                    )
                    attn_radio = gr.Radio(
                        choices=["On", "Off"],
                        value="Off",
                        show_label=False,
                        elem_classes=["pill-radio"],
                    )

                    gr.Markdown(
                        '<div class="field-label">Attention video <span class="hint">one frame per word, playable</span></div>'
                    )
                    video_radio = gr.Radio(
                        choices=["On", "Off"],
                        value="Off",
                        show_label=False,
                        elem_classes=["pill-radio"],
                    )

                    generate_btn = gr.Button("Generate caption", elem_classes=["btn-generate"])

                with gr.Column(scale=6, elem_classes=["glass-card", "panel-right"]):
                    gr.Markdown("Caption", elem_classes=["panel-eyebrow"])
                    caption_out = gr.Markdown(
                        "Upload an image and press *Generate caption*.",
                        elem_classes=["caption-output", "caption-placeholder"],
                    )
                    meta_out = gr.Markdown("", elem_classes=["meta-line"])
                    attention_out = gr.Image(
                        label="Attention over 49 encoder patches",
                        visible=False,
                        elem_classes=["attention-panel"],
                    )
                    video_out = gr.Video(
                        label="Attention video — one frame per generated word",
                        visible=False,
                        elem_id="attn-video",
                        elem_classes=["attention-panel", "video-panel"],
                        autoplay=True,
                        loop=True,
                    )
                    with gr.Row(elem_classes=["speed-row"], visible=True):
                        gr.Markdown('<span class="hint">playback speed</span>', elem_classes=["speed-label"])
                        speed_radio = gr.Radio(
                            choices=["0.25x", "0.5x", "1x", "1.5x", "2x"],
                            value="1x",
                            show_label=False,
                            elem_classes=["pill-radio", "speed-radio"],
                        )

            if EXAMPLE_IMAGES:
                gr.Markdown("Try one of these", elem_classes=["examples-eyebrow"])
                gr.Examples(
                    examples=[[str(APP_DIR / p)] for p in EXAMPLE_IMAGES],
                    inputs=[image_in],
                    outputs=[image_in, source_radio, upload_group, url_group],
                    fn=lambda img: (img, "Upload", gr.Group(visible=True), gr.Group(visible=False)),
                    label=None,
                )

            footer_text = _footer_readout()
            if footer_text:
                gr.Markdown(footer_text, elem_classes=["footer-readout"])

            source_radio.change(
                fn=toggle_source_mode,
                inputs=[source_radio],
                outputs=[upload_group, url_group],
            )

            url_load_btn.click(fn=load_url_preview, inputs=[url_box], outputs=[url_preview])

            dict_toggle_btn.click(
                fn=toggle_dictionary,
                inputs=[dict_state],
                outputs=[dict_state, dict_panel, dict_toggle_btn],
            )
            # .input() fires on every keystroke (not just on blur/submit), so
            # results update live as you type — "y" then "ye" narrows further.
            dict_search.input(fn=search_vocab, inputs=[dict_search], outputs=[dict_results])

            compare_btn.click(
                fn=compare_words,
                inputs=[word_a_box, word_b_box],
                outputs=[embed_results, embed_plot],
            )

            generate_btn.click(fn=None, js=ADD_SCANNING_JS).then(
                fn=generate_caption,
                inputs=[source_radio, image_in, url_box, url_preview, beam_radio, attn_radio, video_radio],
                outputs=[caption_out, meta_out, attention_out, video_out],
            ).then(fn=None, js=REMOVE_SCANNING_JS).then(fn=None, inputs=[speed_radio], js=APPLY_SPEED_JS)

            # Changing speed while a video is already showing takes effect immediately —
            # this is just video.playbackRate on the existing element, no re-encoding.
            speed_radio.change(fn=None, inputs=[speed_radio], js=APPLY_SPEED_JS)

    return demo


def parse_args():
    p = argparse.ArgumentParser(description="Launch the image-captioning UI")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint (default: auto-detect)")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint or args.device:
        try:
            load_model(args.checkpoint, args.device)
        except Exception as e:
            print(f"[warn] could not preload model: {e}")

    demo = build_app()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()