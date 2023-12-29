import logging

import sklearn.base
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import xarray as xr

import a6.evaluation.evaluate as evaluate
import a6.evaluation.metrics as metrics
import a6.types as types
import a6.utils as utils
import mlflow

logger = logging.getLogger(__name__)


def fit(
    model: types.Model,
    X: xr.DataArray,  # noqa: N803
    y: xr.DataArray,
) -> sklearn.base.RegressorMixin:
    """Train a given model."""
    return model.fit(X=utils.transpose(X), y=utils.transpose(y))  # noqa: N803


def train(
    model: nn.Module,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    log_to_mlflow: bool = True,
) -> None:
    loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            # Copy data to GPU
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward the input images through the model
            outputs = model(images)

            # Calculate loss and gradients backwards
            loss = loss_fn(outputs, targets)
            loss.backward()
            running_loss += loss.item()

            # Use the gradients to adjust/optimize the parameters
            optimizer.step()

            # Print every 200 batches
            if i == 0 or i % 200 == 199:
                accuracy = metrics.accuracy_score(
                    y_true=targets, y_pred=outputs
                )
                logger.info(
                    "[EPOCH %i, ITERATION %i] loss=%.4f, accuracy=%.4f",
                    epoch,
                    i,
                    loss.item(),
                    accuracy,
                )

        running_loss /= len(train_loader)

        logger.info(
            "[EPOCH %i] loss_avg=%.4f, accuracy=%.4f",
            epoch,
            running_loss,
            accuracy,
        )

        test_loss, test_accuracy = evaluate.evaluate(
            model,
            epoch=epoch,
            loss_fn=loss_fn,
            test_loader=test_loader,
            device=device,
        )

        if log_to_mlflow:
            utils.mantik.call_mlflow_method(
                mlflow.log_metrics,
                {
                    "train_loss": running_loss,
                    "train_accuracy": accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
                step=epoch,
            )

    logger.info("Finished Training")
