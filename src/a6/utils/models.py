import torch.nn as nn


def get_number_of_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_number_of_non_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)
