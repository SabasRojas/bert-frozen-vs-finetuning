import argparse
import csv
import os
import random
from typing import Dict, List, Tuple

import torch

from code.config import TrainConfig
from code.data_utils import build_dataloaders
from code.model_utils import build_model
from code.trainer import train_model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train frozen BERT baseline or full fine-tuning on AG News."
    )
    parser.add_argument("--dataset", choices=["ag_news"], default="ag_news")
    parser.add_argument(
        "--mode",
        choices=["frozen", "finetune", "both"],
        default="both",
        help="'frozen' = linear probe only, 'finetune' = full BERT, 'both' = run and compare",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_steps", type=int, default=20)
    args = parser.parse_args()

    return TrainConfig(
        dataset_name=args.dataset,
        mode=args.mode,
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


def run_experiment(
    config: TrainConfig,
    train_loader,
    test_loader,
    num_labels: int,
    freeze_bert: bool,
) -> Tuple[List[Dict], float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_label = "frozen" if freeze_bert else "finetune"

    model = build_model(
        model_name=config.model_name,
        num_labels=num_labels,
        freeze_bert=freeze_bert,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{mode_label}] trainable params: {trainable:,} / {total:,}")

    loss_log, test_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        log_every_steps=config.log_every_steps,
        mode_label=mode_label,
    )

    return loss_log, test_acc


def save_results(
    results: List[Dict],
    loss_logs: Dict[str, List[Dict]],
    out_dir: str = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── summary table ──────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "comparison_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "final_test_acc"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved summary  → {summary_path}")

    # ── per-mode loss logs ─────────────────────────────────────────────────
    for mode_label, log in loss_logs.items():
        log_path = os.path.join(out_dir, f"loss_log_{mode_label}.csv")
        if not log:
            continue
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log[0].keys())
            writer.writeheader()
            writer.writerows(log)
        print(f"Saved loss log → {log_path}")


def main() -> None:
    config = parse_args()
    set_seed(config.random_seed)

    print(f"\nLoading dataset: {config.dataset_name}")
    train_loader, test_loader, num_labels = build_dataloaders(
        dataset_name=config.dataset_name,
        tokenizer_name=config.model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    summary_rows: List[Dict] = []
    loss_logs: Dict[str, List[Dict]] = {}

    run_frozen = config.mode in ("frozen", "both")
    run_finetune = config.mode in ("finetune", "both")

    if run_frozen:
        log, acc = run_experiment(config, train_loader, test_loader, num_labels, freeze_bert=True)
        summary_rows.append({"mode": "frozen", "final_test_acc": round(acc, 4)})
        loss_logs["frozen"] = log

    if run_finetune:
        log, acc = run_experiment(config, train_loader, test_loader, num_labels, freeze_bert=False)
        summary_rows.append({"mode": "finetune", "final_test_acc": round(acc, 4)})
        loss_logs["finetune"] = log

    save_results(summary_rows, loss_logs)

    print("\n=== Final comparison ===")
    for row in summary_rows:
        print(f"  {row['mode']:>10}  test_acc={row['final_test_acc']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
