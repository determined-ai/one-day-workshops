from typing import Any, Dict

import torch
from torch import nn

from determined import pytorch


class Flatten(nn.Module):
    def forward(self, *args: pytorch.TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        return x.contiguous().view(x.size(0), -1)


def build_model(n_filters1, n_filters2, dropout1, dropout2) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, n_filters1, 3, 1),
        nn.ReLU(),
        nn.Conv2d(
            n_filters1,
            n_filters2,
            3,
        ),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout1),
        Flatten(),
        nn.Linear(144 * n_filters2, 128),
        nn.ReLU(),
        nn.Dropout2d(dropout2),
        nn.Linear(128, 10),
        nn.LogSoftmax(),
    )
