# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import enum
from collections.abc import Callable

import torch
import torch.nn as nn

import a6.models.resnet as _resnet


class ResNet(_resnet.ResNet):
    def __init__(
        self,
        block,
        layers,
        device: torch.device | None = None,
        in_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: _resnet.BatchNormalization | None = None,
        normalize: bool = False,
        output_dim: int = 0,
        hidden_mlp: int = 0,
        nmb_prototypes: int = 0,
        eval_mode: bool = False,
    ):
        super().__init__(
            block=block,
            layers=layers,
            in_channels=in_channels,
            # Output layer will be overridden
            n_classes=0,
            zero_init_residual=zero_init_residual,
            groups=groups,
            widen=widen,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

        self.eval_mode = eval_mode
        self.device = device
        # normalize output features
        self.l2norm = normalize

        # Override output layer
        self.fc = None

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Sequential(
                nn.Linear(
                    self.num_final_output_filters * block.expansion, output_dim
                ),
                nn.LogSoftmax(dim=1),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(
                    self.num_final_output_filters * block.expansion, hidden_mlp
                ),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        else:
            self.prototypes = None

        self._prepare(zero_init_residual=zero_init_residual)

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_head(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(
                torch.cat(inputs[start_idx:end_idx]).to(
                    device=self.device, non_blocking=True
                )
            )
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim: int, nmb_prototypes: list[int]):
        super().__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module(
                f"prototypes{i}", nn.Linear(output_dim, k, bias=False)
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            getattr(self, "prototypes" + str(i))(x)
            for i in range(self.nmb_heads)
        ]


def resnet50(**kwargs):
    return ResNet(_resnet.Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
    return ResNet(_resnet.Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
    return ResNet(_resnet.Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


def resnet50w5(**kwargs):
    return ResNet(_resnet.Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)


class Architecture(enum.StrEnum):
    ResNet50 = "resnet50"
    ResNet50W2 = "resnet50w2"
    ResNet50W4 = "resnet50w4"
    ResNet50W5 = "resnet50w5"


Models: dict[str, Callable] = {
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
    "resnet50w5": resnet50w5,
}
