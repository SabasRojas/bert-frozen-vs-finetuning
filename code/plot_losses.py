# quick plot of loss_log_frozen.csv and loss_log_finetune.csv -> results/loss_curves.png
# from repo root:  python3 -m code.plot_losses

import csv
import os
from pathlib import Path
from typing import List, Tuple

_root = Path(__file__).resolve().parents[1]
_mpl_dir = _root / ".mplconfig"
_mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_loss_csv(path: Path) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["train_loss"]))
    return steps, losses


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    plt.figure(figsize=(8, 5))
    for name, leg in [("frozen", "frozen"), ("finetune", "finetune")]:
        path = results / f"loss_log_{name}.csv"
        if not path.exists():
            continue
        steps, losses = _load_loss_csv(path)
        plt.plot(steps, losses, label=leg, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title("AG News train loss")
    plt.legend()
    plt.tight_layout()
    out = results / "loss_curves.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
