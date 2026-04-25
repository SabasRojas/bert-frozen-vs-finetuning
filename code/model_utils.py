import torch
from transformers import AutoModelForSequenceClassification


def build_model(model_name: str, num_labels: int, freeze_bert: bool) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if freeze_bert:
        for name, parameter in model.named_parameters():
            if name.startswith("bert."):
                parameter.requires_grad = False

    return model


def trainable_parameters(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
