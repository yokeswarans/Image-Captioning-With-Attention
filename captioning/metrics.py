from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from .vocab import Vocabulary


def _ensure_nltk():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


_ensure_nltk()

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

try:
    from pycocoevalcap.cider.cider import Cider

    _cider_scorer = Cider()
    HAS_CIDER = True
except Exception:
    HAS_CIDER = False


@torch.no_grad()
def generate_greedy(encoder, decoder, encoder_out, vocab, device, encoder_dim, max_len=50):
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    h, c = decoder.init_hidden_state(encoder_out)
    word = torch.tensor([vocab.word2idx[Vocabulary.START]], device=device)
    words, alphas = [], []
    for _ in range(max_len):
        embed = decoder.embedding(word)
        context, alpha = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context
        h, c = decoder.lstm_cell(torch.cat([embed, context], dim=1), (h, c))
        preds = decoder.fc(h)
        word = preds.argmax(dim=1)
        alphas.append(alpha.squeeze(0).cpu().numpy())
        if word.item() == vocab.word2idx[Vocabulary.END]:
            break
        words.append(word.item())
    return words, alphas


@torch.no_grad()
def compute_val_loss(encoder, decoder, loader, criterion, device, amp=True, max_batches=None):
    encoder.eval()
    decoder.eval()
    running_loss, n_batches = 0.0, 0
    for bi, (images, captions, lengths, _) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        with autocast(device.type, enabled=amp and device.type == "cuda"):
            encoder_out = encoder(images)
            predictions, alphas, decode_lengths = decoder(encoder_out, captions, lengths, teacher_forcing_ratio=1.0)
            targets = captions[:, 1:]
            packed_preds = torch.cat(
                [predictions[b, : decode_lengths[b], :] for b in range(len(decode_lengths))], dim=0
            )
            packed_targets = torch.cat(
                [targets[b, : decode_lengths[b]] for b in range(len(decode_lengths))], dim=0
            )
            loss = criterion(packed_preds, packed_targets)
            loss = loss + 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(encoder, decoder, loader, vocab, cfg, device, max_batches=None):
    encoder.eval()
    decoder.eval()
    refs_by_id, hyps_by_id = {}, {}

    for bi, (images, captions, lengths, img_ids) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        images = images.to(device)
        enc_out = encoder(images)
        for i in range(images.size(0)):
            words, _ = generate_greedy(
                encoder, decoder, enc_out[i : i + 1], vocab, device, cfg.encoder_dim, cfg.max_caption_len
            )
            hyp = vocab.decode(words, join=False)
            img_id = img_ids[i]
            hyps_by_id.setdefault(img_id, hyp)
            gt_tokens = vocab.decode(captions[i].tolist(), join=False)
            refs_by_id.setdefault(img_id, []).append(gt_tokens)

    bleu1_l, bleu2_l, bleu3_l, bleu4_l, meteor_l, rougeL_l = [], [], [], [], [], []
    for img_id, hyp in hyps_by_id.items():
        refs = refs_by_id[img_id]
        if len(hyp) == 0:
            continue
        bleu1_l.append(sentence_bleu(refs, hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu2_l.append(sentence_bleu(refs, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu3_l.append(sentence_bleu(refs, hyp, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothie))
        bleu4_l.append(sentence_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
        try:
            meteor_l.append(meteor_score(refs, hyp))
        except Exception:
            pass
        hyp_str, ref_str = " ".join(hyp), " ".join(refs[0])
        rougeL_l.append(rouge.score(ref_str, hyp_str)["rougeL"].fmeasure)

    metrics = {
        "BLEU-1": float(np.mean(bleu1_l)) if bleu1_l else 0.0,
        "BLEU-2": float(np.mean(bleu2_l)) if bleu2_l else 0.0,
        "BLEU-3": float(np.mean(bleu3_l)) if bleu3_l else 0.0,
        "BLEU-4": float(np.mean(bleu4_l)) if bleu4_l else 0.0,
        "METEOR": float(np.mean(meteor_l)) if meteor_l else 0.0,
        "ROUGE-L": float(np.mean(rougeL_l)) if rougeL_l else 0.0,
    }

    if HAS_CIDER:
        gts = {str(k): [" ".join(r) for r in v] for k, v in refs_by_id.items()}
        res = {str(k): [" ".join(v)] for k, v in hyps_by_id.items()}
        cider_score, _ = _cider_scorer.compute_score(gts, res)
        metrics["CIDEr-D"] = float(cider_score)

    return metrics
