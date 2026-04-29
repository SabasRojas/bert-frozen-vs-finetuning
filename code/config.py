from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "bert-base-uncased"
    dataset_name: str = "ag_news"
    mode: str = "frozen"          # "frozen" | "finetune" | "both"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    random_seed: int = 42
    log_every_steps: int = 20
