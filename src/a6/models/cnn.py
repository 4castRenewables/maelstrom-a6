import torch
import torch.nn as nn


class TestingModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        example: torch.Tensor,
        output_dim: int = 2048,
    ):
        super().__init__()
        out_channels_1st_layer = in_channels * 6
        out_channels_2nd_layer = out_channels_1st_layer * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels_1st_layer,
                kernel_size=2,
            ),
            nn.ELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_1st_layer,
                out_channels=out_channels_2nd_layer,
                kernel_size=2,
            ),
            nn.ELU(),
        )
        in_features_mlp = self.conv2(self.conv1(example)).flatten().size(0)
        self.mlp = nn.Sequential(
            # Flatten all dimensions except batch (`start_dim=1`).
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_features_mlp, out_features=output_dim),
            nn.ELU(),
            nn.Linear(in_features=output_dim, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mlp(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        example: torch.Tensor,
        output_dim: int = 2048,
    ):
        super().__init__()
        out_channels_1st_layer = in_channels * 6
        out_channels_2nd_layer = out_channels_1st_layer * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels_1st_layer,
                kernel_size=7,
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_1st_layer,
                out_channels=out_channels_2nd_layer,
                kernel_size=7,
            ),
            nn.ELU(),
            nn.MaxPool2d(2),
        )
        in_features_mlp = self.conv2(self.conv1(example)).flatten().size(0)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features_mlp, out_features=output_dim),
            nn.ELU(),
            nn.Linear(in_features=output_dim, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mlp(x)
        return x
