import argparse
import random

import torch

from code.config import TrainConfig
from code.data_utils import build_dataloaders
from code.model_utils import build_model
from code.trainer import train_model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train frozen BERT baseline on AG News.")
    parser.add_argument("--dataset", choices=["ag_news"], default="ag_news")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_steps", type=int, default=20)
    args = parser.parse_args()

    return TrainConfig(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        random_seed=args.seed,
        log_every_steps=args.log_every_steps,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_frozen_baseline(config: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, num_labels = build_dataloaders(
        dataset_name=config.dataset_name,
        tokenizer_name=config.model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    model = build_model(
        model_name=config.model_name,
        num_labels=num_labels,
        freeze_bert=True,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        log_every_steps=config.log_every_steps,
        mode_label="frozen_baseline",
    )


def main() -> None:
    config = parse_args()
    set_seed(config.random_seed)

    run_frozen_baseline(config)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
