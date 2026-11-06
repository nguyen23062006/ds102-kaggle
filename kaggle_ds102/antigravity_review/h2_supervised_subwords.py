"""Hypothesis 2: Supervised Subword / Character-Ngram CNN from Scratch.

Addresses the representation bottleneck: replaces weak unsupervised skipgram
(V56/V57) with an end-to-end supervised subword convolutional neural network
trained directly on the task labels with class-balanced focal loss.

Constraints:
- No pretrained models or weights (all embeddings and filters randomly initialized).
- No external data.
- Open-source PyTorch / NumPy implementation.
- Guarded: training strictly restricted to hosted Google Colab.
"""

import os
import platform
import re
from collections import Counter
import numpy as np


def guard():
    """Ensure training/fitting only occurs in hosted Colab environments."""
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')


def build_char_ngram_vocab(texts, min_n=3, max_n=5, max_vocab=20000, min_freq=3):
    """Build character n-gram vocabulary from training texts only."""
    counter = Counter()
    for text in texts:
        clean = re.sub(r'\s+', ' ', str(text).lower().strip())
        padded = f'^{clean}$'
        length = len(padded)
        for n in range(min_n, max_n + 1):
            for i in range(length - n + 1):
                counter[padded[i:i + n]] += 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ng, count in counter.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[ng] = len(vocab)
    return vocab


def text_to_ngram_indices(text, vocab, min_n=3, max_n=5, max_len=256):
    """Convert a text string to fixed-length character n-gram index sequence."""
    clean = re.sub(r'\s+', ' ', str(text).lower().strip())
    padded = f'^{clean}$'
    indices = []
    length = len(padded)
    for n in range(min_n, max_n + 1):
        for i in range(length - n + 1):
            ng = padded[i:i + n]
            indices.append(vocab.get(ng, 1))  # 1 is <UNK>
            if len(indices) >= max_len:
                break
        if len(indices) >= max_len:
            break

    # Pad or truncate
    if len(indices) < max_len:
        indices.extend([0] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]
    return indices


class PureNumpyCNNForward:
    """Numpy-only forward pass for dry-run verification and lightweight inference."""

    def __init__(self, vocab_size=5000, embed_dim=32, num_classes=4):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        # Random weights for structural verification
        rng = np.random.RandomState(102)
        self.embeddings = rng.randn(vocab_size, embed_dim).astype(np.float32) * 0.05
        self.conv_w = rng.randn(3, embed_dim, 32).astype(np.float32) * 0.05  # kernel 3, in_c, out_c
        self.fc_w = rng.randn(32, num_classes).astype(np.float32) * 0.05
        self.fc_b = np.zeros(num_classes, dtype=np.float32)

    def forward(self, input_indices):
        """Forward pass on batch of indices (B, T)."""
        B, T = input_indices.shape
        # Embedding lookup: (B, T, D)
        x = self.embeddings[input_indices]
        # 1D conv with kernel size 3
        K = 3
        out_channels = 32
        out_len = max(T - K + 1, 1)
        conv_out = np.zeros((B, out_len, out_channels), dtype=np.float32)
        for t in range(out_len):
            patch = x[:, t:t + K, :]  # (B, K, D)
            # einsum: B K D, K D C -> B C
            conv_out[:, t, :] = np.tensordot(patch, self.conv_w, axes=([1, 2], [0, 1]))
        # ReLU
        conv_out = np.maximum(conv_out, 0)
        # Global Max Pooling: (B, out_channels)
        pooled = np.max(conv_out, axis=1)
        # FC
        logits = np.dot(pooled, self.fc_w) + self.fc_b
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return probs


def get_colab_pytorch_module_code():
    """Return PyTorch model architecture code to be executed on hosted Colab."""
    return '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubwordCNNClassifier(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=64, num_filters=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x).transpose(1, 2)  # (B, D, T)
        c3 = F.relu(self.conv3(emb))
        c4 = F.relu(self.conv4(emb))
        c5 = F.relu(self.conv5(emb))
        p3 = F.adaptive_max_pool1d(c3, 1).squeeze(-1)
        p4 = F.adaptive_max_pool1d(c4, 1).squeeze(-1)
        p5 = F.adaptive_max_pool1d(c5, 1).squeeze(-1)
        pooled = torch.cat([p3, p4, p5], dim=-1)
        logits = self.fc(self.dropout(pooled))
        return logits

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weights = class_weights

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        ce = F.nll_loss(log_p, targets, weight=self.weights, reduction='none')
        pt = p.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce
        return focal_loss.mean()
'''


if __name__ == '__main__':
    print("Testing Hypothesis 2: Supervised Subwords...")
    sample_texts = [
        "Hàng đóng gói rất đẹp, người bán có tâm.",
        "Máy giặt dùng rất êm, ít nhăn quần áo.",
        "Chưa dùng nên chưa biết chất lượng thế nào.",
        "Tb khuyến mãi 3gb data xem bóng đá."
    ]
    vocab = build_char_ngram_vocab(sample_texts, min_n=3, max_n=4, max_vocab=500, min_freq=1)
    print(f"Built character n-gram vocab with {len(vocab)} tokens.")
    batch_indices = np.array([text_to_ngram_indices(t, vocab, max_len=32) for t in sample_texts])
    print("Batch indices shape:", batch_indices.shape)
    
    forward_runner = PureNumpyCNNForward(vocab_size=len(vocab), embed_dim=16, num_classes=4)
    probs = forward_runner.forward(batch_indices)
    print("Probabilities shape:", probs.shape)
    assert probs.shape == (4, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
    print("Hypothesis 2 local verification passed!")
