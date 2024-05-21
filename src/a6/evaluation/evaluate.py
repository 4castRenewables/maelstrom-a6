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
        running_loss = 0.0
        running_accuracy = 0.0

        for i, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, targets).item()
            running_loss += loss

            accuracy = metrics.accuracy_score(y_true=targets, y_pred=outputs)
            running_accuracy += accuracy

            # Print every 50 batches
            if i == 0 or i % 50 == 49:
                accuracy = metrics.accuracy_score(
                    y_true=targets, y_pred=outputs
                )
                logger.info(
                    (
                        "[EPOCH %i, ITERATION %i] "
                        "test_loss=%.4f, test_accuracy: %.4f"
                    ),
                    epoch,
                    i,
                    loss,
                    accuracy,
                )

        running_loss /= len(test_loader)
        running_accuracy /= len(test_loader)

        logger.info(
            "[EPOCH %i] test_loss_avg=%.4f, test_accuracy_avg=%.4f",
            epoch,
            running_loss,
            running_accuracy,
        )

    return running_loss, running_accuracy
