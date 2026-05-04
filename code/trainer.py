from typing import Dict, List, Tuple

import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from tqdm.auto import tqdm

from code.data_utils import move_batch_to_device


def train_model(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    log_every_steps: int,
    mode_label: str,
) -> Tuple[List[Dict], float, float]:
    # returns loss rows, last epoch acc, last epoch weighted f1
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.to(device)

    loss_log: List[Dict] = []

    print(f"\n--- {mode_label} ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"{mode_label} | epoch {epoch + 1}/{epochs}", leave=False)

        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            epoch_loss += loss_value

            if step % log_every_steps == 0:
                entry = {"epoch": epoch + 1, "step": step, "train_loss": round(loss_value, 4)}
                loss_log.append(entry)
                print(f"[{mode_label}] epoch={epoch + 1} step={step} train_loss={loss_value:.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"[{mode_label}] epoch={epoch + 1} avg_train_loss={avg_loss:.4f}")

        test_acc, test_f1 = _evaluate(model, test_loader, device, mode_label, epoch, epochs)

    return loss_log, test_acc, test_f1


def _evaluate(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    mode_label: str,
    epoch: int,
    epochs: int,
) -> Tuple[float, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{mode_label} | eval epoch {epoch + 1}/{epochs}", leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    total = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total if total > 0 else 0.0
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    print(f"[{mode_label}] epoch={epoch + 1} test_acc={acc:.4f} test_f1_weighted={f1:.4f}")
    return acc, f1
