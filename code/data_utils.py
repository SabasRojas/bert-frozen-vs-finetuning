from typing import Dict, Tuple

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


def _dataset_spec(dataset_name: str) -> Tuple[str, str | None, str, str]:
    """
    Returns:
        (dataset_path, dataset_config, text_key, label_key)
    """
    if dataset_name == "ag_news":
        return "ag_news", None, "text", "label"
    raise ValueError("Only ag_news is wired in this phase.")


def _tokenize_dataset(
    raw_dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    text_key: str,
    max_length: int,
) -> DatasetDict:
    def tokenize_batch(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(
            batch[text_key],
            truncation=True,
            max_length=max_length,
        )

    tokenized = raw_dataset.map(tokenize_batch, batched=True)
    tokenized = tokenized.remove_columns(
        [col for col in tokenized["train"].column_names if col not in {"input_ids", "attention_mask", "label"}]
    )
    return tokenized


def build_dataloaders(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int]:
    """Returns (train_loader, test_loader, num_labels)."""
    dataset_path, dataset_config, text_key, label_key = _dataset_spec(dataset_name)
    raw = load_dataset(dataset_path, dataset_config) if dataset_config else load_dataset(dataset_path)

    if label_key != "label":
        raw = raw.rename_column(label_key, "label")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized = _tokenize_dataset(raw, tokenizer, text_key, max_length)

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        tokenized["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    num_labels = int(raw["train"].features["label"].num_classes)
    return train_loader, test_loader, num_labels


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
