import os
from typing import Any, Dict, Sequence, Tuple, Union, cast, List
import logging

import torch
from torch import nn
from determined.pytorch import DataLoader, PyTorchTrial
from torchvision import models, transforms
import numpy as np
from PIL import Image

from data import CatDogDataset
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# =============================================================================

class DogCatModel(PyTorchTrial):
    def __init__(self, context):
        self.context = context

        files = self.download_data()
        self.create_datasets(files)

        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=float(self.context.get_hparam("learning_rate")),
                                    momentum=0.9,
                                    weight_decay=float(self.context.get_hparam("weight_decay")),
                                    nesterov=self.context.get_hparam("nesterov"))

        self.model = self.context.wrap_model(model)
        self.optimizer = self.context.wrap_optimizer(optimizer)
        self.labels = ['dog', 'cat']

    # -------------------------------------------------------------------------

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        loss = torch.nn.functional.cross_entropy(output, labels)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

        return {"loss": loss, "train_accuracy": accuracy}

    # -------------------------------------------------------------------------

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        """
        Calculate validation metrics for a batch and return them as a dictionary.
        This method is not necessary if the user overwrites evaluate_full_dataset().
        """
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch
        output = self.model(data)

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

        loss = torch.nn.functional.cross_entropy(output, labels)

        return {"accuracy": accuracy, "val_loss": loss}

    # -------------------------------------------------------------------------

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.context.get_per_slot_batch_size())

    # -------------------------------------------------------------------------

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.context.get_per_slot_batch_size())

    # -------------------------------------------------------------------------

    def download_data(self):
        data_config = self.context.get_data_config()
        data_dir    = data_config['dir']

        files = []
        for file in os.listdir(data_dir):
            files.append(os.path.join(data_dir, file))

        print(f'Data dir {data_dir} contains {len(files)} files')
        return files

    # -------------------------------------------------------------------------

    def create_datasets(self, files):
        print(f"Creating datasets from {len(files)} input files")
        train_size = round(0.95 * len(files))
        val_size   = len(files) - train_size
        train_ds, val_ds = torch.utils.data.random_split(files, [train_size, val_size])

        self.train_ds = CatDogDataset(train_ds, transform=self.get_train_transforms())
        self.val_ds   = CatDogDataset(val_ds,   transform=self.get_test_transforms())
        print(f"Datasets created: train_size={train_size}, val_size={val_size}")

    # -------------------------------------------------------------------------

    def get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(240),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # -------------------------------------------------------------------------

    def get_test_transforms(self):
        return transforms.Compose([
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # -------------------------------------------------------------------------

    def predict(self, X: np.ndarray, names, meta) -> Union[np.ndarray, List, str, bytes, Dict]:

        image = Image.fromarray(X.astype(np.uint8))
        logging.info(f"Image size : {image.size}")

        image = self.get_test_transforms()(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)[0]
            pred = np.argmax(output)
            logging.info(f"Prediction is : {pred}")

        return [self.labels[pred]]

# =============================================================================
