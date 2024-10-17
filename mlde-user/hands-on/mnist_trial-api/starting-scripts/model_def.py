"""
This example shows how to interact with the Determined PyTorch training APIs to
build a basic MNIST network.

In the `__init__` method, the model and optimizer are wrapped with `wrap_model`
and `wrap_optimizer`. This model is single-input and single-output.

The methods `train_batch` and `evaluate_batch` define the forward pass
for training and evaluation respectively.

Then, configure and run the training loop with the PyTorch Trainer API.
The model can be trained either locally or on-cluster with the same training code.

"""
import pathlib
from typing import Any, Dict

import data # We'll use a function defined in data.py to fetch the data we'll train on.
import torch
import model # The model we'll use is stored in model.py. Have a look!
from ruamel import yaml
from torch import nn

import determined as det
from determined import pytorch


class MNistTrial(pytorch.PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext) -> None:
        # The context is passed from the superclass PyTorchTrial
        self.context = context

        # We set sine trial-level constants. Values could also be sourced from the experiment config.
        self.data_dir = pathlib.Path("data")
        self.batch_size = 64
        self.per_slot_batch_size = self.batch_size // self.context.distributed.get_size()

        # Define loss function.
        self.loss_fn = nn.NLLLoss()

        pass

    def build_training_data_loader(self) -> pytorch.DataLoader:
        train_data = data.get_dataset(self.data_dir, train=True)
        pass

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        validation_data = data.get_dataset(self.data_dir, train=False)
        pass

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        pass

    def evaluate_batch(self, batch: pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        pass
