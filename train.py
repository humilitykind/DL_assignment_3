"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

import wandb

from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            ignore = target == self.pad_idx
            target_clamped = target.clone()
            target_clamped[ignore] = 0
            true_dist.scatter_(1, target_clamped.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[ignore] = 0
        loss = self.kl_div(log_probs, true_dist)
        denom = (~ignore).sum().clamp(min=1)
        return loss / denom


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    pad_idx = getattr(loss_fn, "pad_idx", 1)

    for step, batch in enumerate(data_iter):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        num_tokens = (tgt_out != pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return 0.0
    return total_loss / total_tokens


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], device=device, dtype=torch.long)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    try:
        import evaluate
        bleu_metric = evaluate.load("bleu")
    except Exception as exc:
        raise ImportError("BLEU evaluation requires the 'evaluate' package.") from exc

    model.eval()
    predictions = []
    references = []

    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = make_src_mask(src)

        for i in range(src.size(0)):
            decoded = greedy_decode(
                model,
                src[i : i + 1],
                src_mask[i : i + 1],
                max_len,
                start_symbol=2,
                end_symbol=3,
                device=device,
            )

            pred_tokens = decoded.squeeze(0).tolist()[1:]
            if pred_tokens and pred_tokens[-1] == 3:
                pred_tokens = pred_tokens[:-1]

            if isinstance(tgt_vocab, dict) and "itos" in tgt_vocab:
                itos = tgt_vocab["itos"]
                pred_text = " ".join([itos[idx] for idx in pred_tokens if idx < len(itos)])
                ref_tokens = tgt[i].tolist()[1:]
                ref_text = " ".join([itos[idx] for idx in ref_tokens if idx not in (1, 3) and idx < len(itos)])
            elif hasattr(tgt_vocab, "lookup_token"):
                pred_text = " ".join([tgt_vocab.lookup_token(idx) for idx in pred_tokens])
                ref_tokens = tgt[i].tolist()[1:]
                ref_text = " ".join([tgt_vocab.lookup_token(idx) for idx in ref_tokens if idx != 1 and idx != 3])
            else:
                pred_text = " ".join([tgt_vocab.itos[idx] for idx in pred_tokens])
                ref_tokens = tgt[i].tolist()[1:]
                ref_text = " ".join([tgt_vocab.itos[idx] for idx in ref_tokens if idx != 1 and idx != 3])

            predictions.append(pred_text)
            references.append([ref_text])

    bleu = bleu_metric.compute(predictions=predictions, references=references)
    return bleu["bleu"] * 100


def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad batch of (src, tgt) sequences to max length in batch."""
    src_batch, tgt_batch = zip(*batch)
    src_max = max(len(x) for x in src_batch)
    tgt_max = max(len(x) for x in tgt_batch)

    src_tensor = torch.full((len(src_batch), src_max), pad_idx, dtype=torch.long)
    tgt_tensor = torch.full((len(tgt_batch), tgt_max), pad_idx, dtype=torch.long)

    for i, (src_seq, tgt_seq) in enumerate(zip(src_batch, tgt_batch)):
        src_tensor[i, : len(src_seq)] = torch.tensor(src_seq, dtype=torch.long)
        tgt_tensor[i, : len(tgt_seq)] = torch.tensor(tgt_seq, dtype=torch.long)

    return src_tensor, tgt_tensor


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    model_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model": model.d_model,
        "N": len(model.encoder.layers),
        "num_heads": model.encoder.layers[0].self_attn.num_heads,
        "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
        "dropout": model.encoder.layers[0].dropout.p,
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    config = {
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "batch_size": 64,
        "num_epochs": 10,
        "lr": 1.0,
        "label_smoothing": 0.1,
        "min_freq": 2,
    }

    wandb.init(entity="arshit1-mankodi-iit-madras", project="DL_3", config=config)
    cfg = wandb.config

    train_ds = Multi30kDataset(split="train", min_freq=cfg.min_freq)
    val_ds = Multi30kDataset(split="validation", min_freq=cfg.min_freq, src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds = Multi30kDataset(split="test", min_freq=cfg.min_freq, src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    pad_idx = train_ds.pad_idx

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab["itos"]),
        tgt_vocab_size=len(train_ds.tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab["itos"]),
        pad_idx=pad_idx,
        smoothing=cfg.label_smoothing,
    )

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt")

    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})


if __name__ == "__main__":
    run_training_experiment()
