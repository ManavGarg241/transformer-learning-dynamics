import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


TOKEN_PATTERN = re.compile(r"\b\w+\b|[.,!?;:]")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_tiny_corpus() -> List[str]:
    # Small handcrafted corpus keeps training lightweight and interpretable.
    return [
        "attention lets each token look at other tokens in a sequence .",
        "transformers use attention and feed forward layers .",
        "a model learns patterns from repeated examples .",
        "sequence models can fail when context is too short .",
        "good optimization helps the loss decrease over time .",
        "layer normalization stabilizes deep learning dynamics .",
        "attention weights reveal which words influence predictions .",
        "multi head attention captures different token relations .",
        "small datasets make failure cases easy to inspect .",
        "learning dynamics can show stable or unstable behavior .",
    ]


def build_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = ["<pad>", "<unk>"] + sorted(set(tokens))
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos


def encode_tokens(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    unk = stoi["<unk>"]
    return [stoi.get(tok, unk) for tok in tokens]


def make_sequences(token_ids: List[int], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(token_ids) - seq_len):
        xs.append(token_ids[i : i + seq_len])
        ys.append(token_ids[i + 1 : i + seq_len + 1])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self.combine_heads(context)
        output = self.out_proj(context)
        return output, attn_weights


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, weights


class TransformerEncoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.embedding(x)
        h = self.pos_encoding(h)

        all_attn = []
        for layer in self.layers:
            h, attn = layer(h)
            all_attn.append(attn)

        logits = self.head(h)
        return logits, all_attn


@dataclass
class Config:
    seq_len: int = 6
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 35
    lr: float = 1e-3


def train_model(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    loss_history = []
    grad_norm_history = []
    model.train()

    for _ in range(cfg.epochs):
        epoch_losses = []
        epoch_grads = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_grads.append(total_norm)

        loss_history.append(float(np.mean(epoch_losses)))
        grad_norm_history.append(float(np.mean(epoch_grads)))

    return loss_history, grad_norm_history


def evaluate_failures(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    itos: Dict[int, str],
    top_k: int = 8,
) -> List[Dict[str, object]]:
    model.eval()
    failures = []
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)

        for i in range(x.size(0)):
            mismatch_positions = (preds[i] != y[i]).nonzero(as_tuple=False).flatten().tolist()
            if mismatch_positions:
                pos = mismatch_positions[0]
                input_tokens = [itos[idx.item()] for idx in x[i]]
                true_tok = itos[y[i, pos].item()]
                pred_tok = itos[preds[i, pos].item()]
                conf = probs[i, pos, preds[i, pos]].item()
                failures.append(
                    {
                        "input": input_tokens,
                        "position": int(pos),
                        "true": true_tok,
                        "pred": pred_tok,
                        "confidence": float(conf),
                    }
                )
            if len(failures) >= top_k:
                break
    return failures


def evaluate_probe_failures(
    model: nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    seq_len: int,
    top_k: int = 8,
) -> List[Dict[str, object]]:
    # Probes intentionally deviate from training distribution to expose brittle behavior.
    probes = [
        "attention in extremely noisy data can become unreliable .",
        "models sometimes predict wrong words when context shifts suddenly .",
        "this sentence contains novel tokens quantum zebra matrix .",
    ]

    failures: List[Dict[str, object]] = []
    model.eval()

    with torch.no_grad():
        for sentence in probes:
            tokens = tokenize(sentence)
            token_ids = encode_tokens(tokens, stoi)
            if len(token_ids) <= seq_len:
                continue

            x_probe, y_probe = make_sequences(token_ids, seq_len)
            logits, _ = model(x_probe)
            preds = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)

            for i in range(x_probe.size(0)):
                mismatch_positions = (preds[i] != y_probe[i]).nonzero(as_tuple=False).flatten().tolist()
                if mismatch_positions:
                    pos = mismatch_positions[0]
                    input_tokens = [itos[idx.item()] for idx in x_probe[i]]
                    true_tok = itos[y_probe[i, pos].item()]
                    pred_tok = itos[preds[i, pos].item()]
                    conf = probs[i, pos, preds[i, pos]].item()
                    failures.append(
                        {
                            "probe_sentence": sentence,
                            "input": input_tokens,
                            "position": int(pos),
                            "true": true_tok,
                            "pred": pred_tok,
                            "confidence": float(conf),
                        }
                    )
                if len(failures) >= top_k:
                    return failures

    return failures


def plot_loss(loss: List[float], grad_norm: List[float], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(loss, color="#2a9d8f", linewidth=2)
    axes[0].set_title("Loss Across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")

    axes[1].plot(grad_norm, color="#e76f51", linewidth=2)
    axes[1].set_title("Gradient Norm Across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L2 Gradient Norm")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_attention_heatmap(attn: torch.Tensor, tokens: List[str], out_path: str) -> None:
    # attn shape: [num_heads, seq, seq] for a single sample and single layer.
    avg_attn = attn.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(7, 6))
    sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Average Attention Weights (Across Heads)")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def prepare_data(seq_len: int):
    corpus = build_tiny_corpus()
    all_tokens = []
    for line in corpus:
        all_tokens.extend(tokenize(line))

    stoi, itos = build_vocab(all_tokens)
    token_ids = encode_tokens(all_tokens, stoi)
    x, y = make_sequences(token_ids, seq_len)
    return x, y, stoi, itos


def run_experiment(cfg: Config, out_dir: str, tag: str) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, stoi, itos = prepare_data(cfg.seq_len)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = TransformerEncoderLM(
        vocab_size=len(stoi),
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)

    loss_history, grad_history = train_model(model, loader, cfg, device)

    model.eval()
    with torch.no_grad():
        logits, all_attn = model(x.to(device))
        final_loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).item()

    train_failures = evaluate_failures(model, x.to(device), y.to(device), itos)
    probe_failures = evaluate_probe_failures(model, stoi, itos, cfg.seq_len)

    loss_plot_path = os.path.join(out_dir, f"{tag}_loss_grad.png")
    plot_loss(loss_history, grad_history, loss_plot_path)

    sample_idx = 0
    sample_tokens = [itos[idx.item()] for idx in x[sample_idx]]
    attn_layer_0_sample = all_attn[0][sample_idx].detach().cpu()
    attn_plot_path = os.path.join(out_dir, f"{tag}_attention_heatmap.png")
    plot_attention_heatmap(attn_layer_0_sample, sample_tokens, attn_plot_path)

    result = {
        "tag": tag,
        "config": cfg.__dict__,
        "final_loss": final_loss,
        "min_train_loss": float(min(loss_history)),
        "max_train_loss": float(max(loss_history)),
        "loss_curve": [float(v) for v in loss_history],
        "grad_norm_curve": [float(v) for v in grad_history],
        "n_train_failures": len(train_failures),
        "n_probe_failures": len(probe_failures),
        "n_failures_captured": len(train_failures) + len(probe_failures),
        "failure_examples_train": train_failures,
        "failure_examples_probe": probe_failures,
        "artifacts": {
            "loss_grad_plot": loss_plot_path,
            "attention_plot": attn_plot_path,
        },
    }

    with open(os.path.join(out_dir, f"{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def run_all() -> None:
    set_seed(42)
    out_dir = "outputs"

    base_cfg = Config()
    experiments = [
        ("base", base_cfg),
        ("heads_2", Config(num_heads=2, d_model=64)),
        ("embed_96", Config(d_model=96, num_heads=4, ff_dim=192)),
        ("seq_8", Config(seq_len=8)),
    ]

    summary = {}
    for tag, cfg in experiments:
        summary[tag] = run_experiment(cfg, out_dir, tag)

    comparison = {
        tag: {
            "final_loss": metrics["final_loss"],
            "min_train_loss": metrics["min_train_loss"],
            "n_failures_captured": metrics["n_failures_captured"],
            "config": metrics["config"],
        }
        for tag, metrics in summary.items()
    }

    with open(os.path.join(out_dir, "experiment_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("Experiments complete. Artifacts saved in outputs/.")


if __name__ == "__main__":
    run_all()
