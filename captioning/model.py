from __future__ import annotations

import random

import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=7, fine_tune=False, pretrained=True):
        super().__init__()
        self.enc_image_size = encoded_image_size
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)
        self.features = backbone.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(fine_tune)

    def forward(self, images):
        feats = self.features(images)
        feats = self.adaptive_pool(feats)
        feats = feats.permute(0, 2, 3, 1)
        return feats

    def fine_tune(self, fine_tune=True):
        for p in self.features.parameters():
            p.requires_grad = False
        if fine_tune:
            for block in list(self.features.children())[-4:]:
                for p in block.parameters():
                    p.requires_grad = True


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.tanh(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)
        h = self.init_h(mean_enc)
        c = self.init_c(mean_enc)
        return h, c

    def forward(self, encoder_out, captions, lengths, teacher_forcing_ratio=1.0):
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (lengths - 1).tolist()
        max_decode_len = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, encoder_out.size(1), device=encoder_out.device)

        embeddings = self.embedding(captions)
        prev_word_embed = embeddings[:, 0, :]

        for t in range(max_decode_len):
            batch_size_t = sum(l > t for l in decode_lengths)
            context, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context

            lstm_input = torch.cat([prev_word_embed[:batch_size_t], context], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            h = torch.cat([h_t, h[batch_size_t:]], dim=0) if batch_size_t < batch_size else h_t
            c = torch.cat([c_t, c[batch_size_t:]], dim=0) if batch_size_t < batch_size else c_t

            preds = self.fc(self.dropout(h_t))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher and t + 1 < embeddings.size(1):
                next_embed = embeddings[:, t + 1, :]
            else:
                next_ids = preds.argmax(dim=1)
                next_embed_t = self.embedding(next_ids)
                if next_embed_t.size(0) < embeddings.size(0):
                    pad = self.embedding(
                        torch.zeros(
                            embeddings.size(0) - next_embed_t.size(0),
                            dtype=torch.long,
                            device=embeddings.device,
                        )
                    )
                    next_embed_t = torch.cat([next_embed_t, pad], dim=0)
                next_embed = next_embed_t
            prev_word_embed = next_embed

        return predictions, alphas, decode_lengths


def build_models(cfg, vocab_size, device, pretrained=True):
    encoder = EncoderCNN(fine_tune=False, pretrained=pretrained).to(device)
    decoder = DecoderWithAttention(
        attention_dim=cfg.attention_dim,
        embed_dim=cfg.embed_dim,
        decoder_dim=cfg.decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=cfg.encoder_dim,
        dropout=cfg.dropout,
    ).to(device)
    return encoder, decoder
