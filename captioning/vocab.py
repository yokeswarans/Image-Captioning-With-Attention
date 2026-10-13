from __future__ import annotations

import json
import os
import pickle
import re
import string
from collections import Counter


PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_caption(caption: str) -> list:
    caption = caption.lower().strip()
    caption = caption.translate(PUNCT_TABLE)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption.split()


def load_coco_annotations(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    id2file = {img["id"]: img["file_name"] for img in data["images"]}
    id2caps = {}
    for ann in data["annotations"]:
        id2caps.setdefault(ann["image_id"], []).append(ann["caption"])
    return id2caps, id2file


class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def build(self, all_tokenized_captions, min_freq=5):
        counter = Counter()
        for toks in all_tokenized_captions:
            counter.update(toks)
        specials = [self.PAD, self.START, self.END, self.UNK]
        words = specials + [w for w, c in counter.items() if c >= min_freq and w not in specials]
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def encode(self, tokens, max_len):
        ids = [self.word2idx.get(self.START)]
        ids += [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens[: max_len - 2]]
        ids.append(self.word2idx[self.END])
        return ids

    def decode(self, ids, join=True):
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), self.UNK)
            if w == self.END:
                break
            if w in (self.START, self.PAD):
                continue
            words.append(w)
        return " ".join(words) if join else words

    def __len__(self):
        return len(self.word2idx)


def build_or_load_vocab_and_captions(ann_path, cache_dir, split_name, min_freq, logger=None):
    log = logger.info if logger else print
    vocab_path = os.path.join(cache_dir, "vocab.pkl")
    caps_path = os.path.join(cache_dir, f"cleaned_captions_{split_name}.pkl")
    paths_path = os.path.join(cache_dir, f"image_paths_{split_name}.pkl")

    if os.path.exists(caps_path) and os.path.exists(paths_path):
        log(f"[cache hit] loading cleaned captions for {split_name}")
        with open(caps_path, "rb") as f:
            id2clean = pickle.load(f)
        with open(paths_path, "rb") as f:
            id2file = pickle.load(f)
    else:
        log(f"[cache miss] cleaning captions for {split_name} ...")
        id2caps, id2file = load_coco_annotations(ann_path)
        id2clean = {img_id: [clean_caption(c) for c in caps] for img_id, caps in id2caps.items()}
        os.makedirs(cache_dir, exist_ok=True)
        with open(caps_path, "wb") as f:
            pickle.dump(id2clean, f)
        with open(paths_path, "wb") as f:
            pickle.dump(id2file, f)

    if split_name == "train":
        if os.path.exists(vocab_path):
            log("[cache hit] loading vocab.pkl")
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
        else:
            log("[cache miss] building vocabulary ...")
            all_tok = [toks for caps in id2clean.values() for toks in caps]
            vocab = Vocabulary().build(all_tok, min_freq=min_freq)
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
        return id2clean, id2file, vocab

    return id2clean, id2file
