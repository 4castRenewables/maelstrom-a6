import logging
from collections.abc import Callable

import torch.nn as nn
import torch.utils.data

import a6.evaluation.metrics as metrics

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    epoch: int,
    loss_fn: Callable,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, targets).item()

            accuracy = metrics.accuracy_score(y_true=targets, y_pred=outputs)

            logger.info(
                "[EPOCH %i] test_loss=%.4f, test_accuracy: %.4f",
                epoch,
                loss,
                accuracy,
            )

    return loss, float(accuracy)
