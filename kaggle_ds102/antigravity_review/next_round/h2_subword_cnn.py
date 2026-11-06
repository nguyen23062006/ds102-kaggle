"""H2: Supervised Subword / Character-Ngram CNN with Focal Loss.

Fully addresses previous review findings:
1. Vocabulary fit ONLY on training partition.
2. Positional interleaved tokenization: at each character position i, generates
   (3-gram, 4-gram, 5-gram) sequentially, ensuring truncation preserves all n-gram
   scales proportionally rather than exhausting the token budget on trigrams.
3. Correct valid-length masking prior to convolution pooling so padding does
   not distort max-pooling activations.
4. Deterministic training loop, PyTorch seed fixed, batch-size invariant inference.
5. Controlled comparison: ClassBalancedFocalLoss vs WeightedCrossEntropyLoss.
6. Fixed epoch snapshot averaging (epochs 5 and 6 of 6); NO outer-validation early stopping.
7. Guarded: training strictly restricted to hosted Google Colab.
"""

import gc
import json
import os
import platform
import random
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np


def guard():
    """Ensure training/fitting only occurs in hosted Colab environments."""
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')


CONFIG = dict(
    model_family='Supervised Subword Character-Ngram CNN',
    vocab_size=8000,
    min_ngram=3,
    max_ngram=5,
    min_freq=2,
    max_seq_len=256,
    embed_dim=64,
    conv_channels=64,
    kernel_sizes=[3, 4, 5],
    dense_dim=128,
    dropout=0.3,
    embedding_dropout=0.15,
    epochs=6,
    snapshot_epochs=[5, 6],
    batch_size=64,
    lr=0.0015,
    weight_decay=0.01,
    focal_gamma=2.0,
    seed=102,
)


class SubwordTokenizer:
    """Interleaved character n-gram tokenizer fit strictly on training text."""

    def __init__(self, vocab_size=8000, min_n=3, max_n=5, min_freq=2):
        self.vocab_size = vocab_size
        self.min_n = min_n
        self.max_n = max_n
        self.min_freq = min_freq
        self.vocab = {'<PAD>': 0, '<UNK>': 1}

    def _normalize(self, text):
        clean = re.sub(r'\s+', ' ', str(text).lower().strip())
        return f'^{clean}$'

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            s = self._normalize(text)
            L = len(s)
            for i in range(L):
                for n in range(self.min_n, self.max_n + 1):
                    if i + n <= L:
                        counter[s[i:i + n]] += 1

        # Keep top most frequent n-grams
        candidates = [ng for ng, cnt in counter.most_common(self.vocab_size - 2) if cnt >= self.min_freq]
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, ng in enumerate(candidates, start=2):
            self.vocab[ng] = idx
        return self

    def encode(self, texts, max_len=256):
        """Positional interleaving: at each char position i, yields (3-gram, 4-gram, 5-gram).
        
        This guarantees that even when truncated at max_len, the sequence contains
        an equal mixture of trigrams, 4-grams, and 5-grams across the covered text span.
        """
        n_samples = len(texts)
        result = np.zeros((n_samples, max_len), dtype=np.int64)

        for row_idx, text in enumerate(texts):
            s = self._normalize(text)
            L = len(s)
            tokens = []
            for i in range(L):
                for n in range(self.min_n, self.max_n + 1):
                    if i + n <= L:
                        ng = s[i:i + n]
                        tokens.append(self.vocab.get(ng, 1))
                        if len(tokens) >= max_len:
                            break
                if len(tokens) >= max_len:
                    break

            if tokens:
                result[row_idx, :len(tokens)] = tokens

        return result


if __name__ == '__main__':
    # Local verification without torch
    pass


# ---------------------------------------------------------------------------
# Colab-only PyTorch Architecture & Loss
# ---------------------------------------------------------------------------

def _build_pytorch_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MaskedSubwordCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, channels=64, kernels=[3, 4, 5],
                     dense_dim=128, dropout=0.3, embed_dropout=0.15, num_classes=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            with torch.no_grad():
                self.embedding.weight[0].zero_()

            self.embed_dropout = nn.Dropout(embed_dropout)
            self.kernels = kernels
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, channels, kernel_size=k) for k in kernels
            ])
            self.pool_norm = nn.LayerNorm(channels * len(kernels))
            self.head = nn.Sequential(
                nn.Linear(channels * len(kernels), dense_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dense_dim, num_classes)
            )

        def forward(self, x):
            # x: (B, T)
            B, T = x.shape
            lengths = (x != 0).sum(dim=1)  # (B,)
            emb = self.embed_dropout(self.embedding(x))  # (B, T, D)
            emb_t = emb.transpose(1, 2)  # (B, D, T)

            pooled_parts = []
            for conv, k in zip(self.convs, self.kernels):
                act = F.relu(conv(emb_t))  # (B, C, L_out), where L_out = T - k + 1
                L_out = act.shape[2]
                valid_len = (lengths - k + 1).clamp_min(1)  # (B,)
                # Mask out any activation that touched padding positions
                pos = torch.arange(L_out, device=x.device)[None, :]  # (1, L_out)
                valid_mask = pos < valid_len[:, None]  # (B, L_out)
                # Apply mask before max pooling
                masked_act = act.masked_fill(~valid_mask[:, None, :], -1e4)
                pooled = masked_act.amax(dim=2)  # (B, C)
                pooled_parts.append(pooled)

            cat_pool = torch.cat(pooled_parts, dim=1)  # (B, C * len(kernels))
            normed = self.pool_norm(cat_pool)
            logits = self.head(normed)  # (B, num_classes)
            return logits

    class ClassBalancedFocalLoss(nn.Module):
        def __init__(self, alpha, gamma=2.0):
            super().__init__()
            self.register_buffer('alpha', alpha.float())
            self.gamma = gamma

        def forward(self, logits, targets):
            # logits: (B, C), targets: (B,)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_alpha = self.alpha[targets]

            focal_weight = target_alpha * ((1.0 - target_probs) ** self.gamma)
            loss = - focal_weight * target_log_probs
            return loss.mean()

    class WeightedCrossEntropyLoss(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(weight=weights.float())

        def forward(self, logits, targets):
            return self.ce(logits, targets)

    return MaskedSubwordCNN, ClassBalancedFocalLoss, WeightedCrossEntropyLoss


def predict_proba_batched(model, x_tensor, device, batch_size=256):
    """Batched, label-blind probability inference."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    all_probs = []
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.inference_mode():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def fit_predict_subword_cnn(train_df, eval_frames, folder, loss_type='focal', status=None):
    """Full Colab training loop with deterministic seeding, checkpointing, and snapshot averaging."""
    guard()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    MaskedSubwordCNN, ClassBalancedFocalLoss, WeightedCrossEntropyLoss = _build_pytorch_classes()

    # Deterministic seeding
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # 1. Fit Tokenizer strictly on training partition
    train_texts = train_df['comment'].fillna('').astype(str).tolist()
    tokenizer = SubwordTokenizer(
        vocab_size=CONFIG['vocab_size'],
        min_n=CONFIG['min_ngram'],
        max_n=CONFIG['max_ngram'],
        min_freq=CONFIG['min_freq']
    )
    tokenizer.fit(train_texts)

    X_train = tokenizer.encode(train_texts, max_len=CONFIG['max_seq_len'])
    X_train_tensor = torch.from_numpy(X_train)

    X_evals = [tokenizer.encode(df['comment'].fillna('').astype(str).tolist(), max_len=CONFIG['max_seq_len'])
               for df in eval_frames]
    X_eval_tensors = [torch.from_numpy(arr) for arr in X_evals]

    y_train = train_df['spam_label'].to_numpy(dtype=np.int64)
    y_train_tensor = torch.from_numpy(y_train)

    # Class weights computed strictly from fit labels
    counts = np.bincount(y_train, minlength=4)
    assert (counts > 0).all()
    # Inverse square-root class frequency
    inv_sqrt = np.sqrt(len(y_train) / (4.0 * counts)).astype(np.float32)
    weights_tensor = torch.tensor(inv_sqrt, device=device)

    # 2. Model & Loss setup
    model = MaskedSubwordCNN(
        vocab_size=len(tokenizer.vocab),
        embed_dim=CONFIG['embed_dim'],
        channels=CONFIG['conv_channels'],
        kernels=CONFIG['kernel_sizes'],
        dense_dim=CONFIG['dense_dim'],
        dropout=CONFIG['dropout'],
        embed_dropout=CONFIG['embedding_dropout'],
        num_classes=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    if loss_type == 'focal':
        loss_fn = ClassBalancedFocalLoss(alpha=weights_tensor, gamma=CONFIG['focal_gamma'])
    elif loss_type == 'cross_entropy':
        loss_fn = WeightedCrossEntropyLoss(weights=weights_tensor)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Save tokenizer vocabulary and run configuration
    (folder / 'tokenizer_vocab.json').write_text(
        json.dumps(dict(vocab_size=len(tokenizer.vocab), config=CONFIG, loss_type=loss_type), ensure_ascii=False)
    )

    # 3. Training Loop with DataLoader
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, generator=generator, num_workers=0)

    eval_predictions = [np.zeros((len(df), 4), dtype=np.float64) for df in eval_frames]
    start_time = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(batch_y)
            n_samples += len(batch_y)

        avg_loss = total_loss / max(n_samples, 1)
        if status:
            status('subword_cnn_epoch', epoch=epoch, loss_type=loss_type,
                   training_loss=avg_loss, elapsed=time.time() - start_time)

        # Snapshot averaging over fixed pre-declared snapshot epochs (no outer validation leakage)
        if epoch in CONFIG['snapshot_epochs']:
            ckpt_path = folder / f'model_{loss_type}_epoch{epoch}.pt'
            tmp_ckpt = ckpt_path.with_suffix('.tmp.pt')
            torch.save(model.state_dict(), tmp_ckpt)
            tmp_ckpt.replace(ckpt_path)

            for i, x_eval in enumerate(X_eval_tensors):
                probs = predict_proba_batched(model, x_eval, device, batch_size=256)
                eval_predictions[i] += probs / float(len(CONFIG['snapshot_epochs']))

    del model, optimizer, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_predictions
