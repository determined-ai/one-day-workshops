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
import logging
import pathlib
from typing import Any, Dict

import data
import model
import torch
from ruamel import yaml
from torch import nn

import determined as det
from determined import pytorch


class MNistTrial(pytorch.PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext) -> None:
        self.context = context

        # Trial-level constants.
        self.data_dir = pathlib.Path("data")
        self.batch_size = 64
        self.per_slot_batch_size = self.batch_size // self.context.distributed.get_size()

        # Define loss function.
        self.loss_fn = nn.NLLLoss()

        # Define model.
        self.model = self.context.wrap_model(
            model.build_model(
                n_filters1=self.context.get_hparam("n_filters1"),
                n_filters2=self.context.get_hparam("n_filters2"),
                dropout1=self.context.get_hparam("dropout1"),
                dropout2=self.context.get_hparam("dropout2"),
            )
        )

        # Configure optimizer.
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adadelta(self.model.parameters(), lr=self.context.get_hparam("learning_rate"))
        )

    def build_training_data_loader(self) -> pytorch.DataLoader:
        train_data = data.get_dataset(self.data_dir, train=True)
        return pytorch.DataLoader(train_data, batch_size=self.per_slot_batch_size)

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        validation_data = data.get_dataset(self.data_dir, train=False)
        return pytorch.DataLoader(validation_data, batch_size=self.per_slot_batch_size)

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch_data, labels = batch

        output = self.model(batch_data)
        loss = self.loss_fn(output, labels)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss}

    def evaluate_batch(self, batch: pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        batch_data, labels = batch

        output = self.model(batch_data)
        validation_loss = self.loss_fn(output, labels).item()

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(batch_data)

        return {"validation_loss": validation_loss, "accuracy": accuracy}
