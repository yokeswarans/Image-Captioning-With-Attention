from __future__ import annotations

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .vocab import Vocabulary


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class CocoCaptionDataset(Dataset):
    def __init__(self, images_dir, id2clean, id2file, vocab: Vocabulary, max_len, transform):
        self.images_dir = images_dir
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform
        self.samples = []
        for img_id, caps in id2clean.items():
            if img_id not in id2file:
                continue
            for toks in caps:
                self.samples.append((img_id, toks))
        self.id2file = id2file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, tokens = self.samples[idx]
        img_path = os.path.join(self.images_dir, self.id2file[img_id])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption_ids = self.vocab.encode(tokens, self.max_len)
        return image, torch.tensor(caption_ids, dtype=torch.long), img_id


def make_collate_fn(vocab: Vocabulary):
    pad_idx = vocab.word2idx[Vocabulary.PAD]

    def collate_fn(batch):
        images, captions, img_ids = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = [len(c) for c in captions]
        max_len = max(lengths)
        padded = torch.full((len(captions), max_len), pad_idx, dtype=torch.long)
        for i, c in enumerate(captions):
            padded[i, : len(c)] = c
        lengths = torch.tensor(lengths, dtype=torch.long)
        return images, padded, lengths, list(img_ids)

    return collate_fn


def exclude_training_images_from_validation(train_id2file, val_id2clean, val_id2file):
    overlap_ids = set(train_id2file).intersection(val_id2file)
    if not overlap_ids:
        return val_id2clean, val_id2file, overlap_ids

    val_id2clean = {
        img_id: captions
        for img_id, captions in val_id2clean.items()
        if img_id not in overlap_ids
    }
    val_id2file = {
        img_id: filename
        for img_id, filename in val_id2file.items()
        if img_id not in overlap_ids
    }
    return val_id2clean, val_id2file, overlap_ids


def build_dataloaders(cfg, train_id2clean, train_id2file, val_id2clean, val_id2file, vocab):
    collate_fn = make_collate_fn(vocab)
    train_dataset = CocoCaptionDataset(
        cfg.train_images, train_id2clean, train_id2file, vocab, cfg.max_caption_len, train_transform
    )
    val_dataset = CocoCaptionDataset(
        cfg.val_images, val_id2clean, val_id2file, vocab, cfg.max_caption_len, eval_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset, val_dataset
