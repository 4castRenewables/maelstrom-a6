import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision

import a6

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print(f"GPU available, using {name}")
    # Declare device to be able to copy model and tensors to GPU
    device = torch.device("cuda")
else:
    print("No GPU available, use CPU instead")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")

# The mean and standard deviation of the dataset
# is 0.1306 and 0.3081, respectively.
normalize = torchvision.transforms.Normalize(mean=0.1306, std=0.3081)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        normalize,
    ]
)

crop_size = 2 / 3
train_transforms = torchvision.transforms.Compose(
    [
        # In original DCv2 paper, the following data
        # augmentation strategies are implemented
        a6.datasets.methods.transform.color_distortion(),
        a6.datasets.methods.transform.PILRandomGaussianBlur(),
        torchvision.transforms.ToTensor(),
        normalize,
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomResizedCrop(
            # MNIST images have size 28x28
            crop_size * 28,
            scale=(0.14, 1.0),
            antialias=True,
        ),
    ]
)

train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transforms,
)
test_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transforms,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    # Evaluate model on a single batch containing all test samples.
    batch_size=len(test_set),
    shuffle=False,
)


def train(
    model: nn.Module, epochs: int
) -> tuple[list[float], list[float], list[float]]:
    loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    train_losses = []
    test_losses = []
    test_accuracies = []

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
            if i % 200 == 199:
                accuracy = a6.evaluation.metrics.accuracy_score(
                    y_true=targets, y_pred=outputs
                )
                print(
                    f"[epoch {epoch}, iteration {i}] "
                    f"loss={loss.item():.4f}, {accuracy=:.4f}",
                )

        running_loss /= len(train_loader)
        train_losses.append(running_loss)

        print(f"[epoch {epoch}, iteration {i}] loss_avg={running_loss:.4f}")

        test_loss, test_accuracy = evaluate(model, epoch=epoch, loss_fn=loss_fn)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    print("Finished Training")

    return train_losses, test_losses, test_accuracies


def evaluate(model: nn.Module, epoch: int, loss_fn) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            # Copy samples to GPU
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, targets).item()

            accuracy = a6.evaluation.metrics.accuracy_score(
                y_true=targets, y_pred=outputs
            )

            print(f"[epoch {epoch}] test {accuracy=:.4f}")

    return loss, float(accuracy)


def main(epochs: int) -> tuple[list[float], list[float], list[float]]:
    model = a6.models.resnet.resnet50(n_classes=10, in_channels=1)
    print(model)

    # Copy model to GPU
    model.to(device)

    return train(model, epochs=epochs)


if __name__ == "__main__":
    main(epochs=10)
