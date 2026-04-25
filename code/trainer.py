import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from code.data_utils import move_batch_to_device


def train_model(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    log_every_steps: int,
    mode_label: str,
) -> None:
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    print(f"\n=== Training mode: {mode_label} ===")
    for epoch in range(epochs):
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
                print(f"[{mode_label}] epoch={epoch + 1} step={step} train_loss={loss_value:.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"[{mode_label}] epoch={epoch + 1} avg_train_loss={avg_loss:.4f}")
