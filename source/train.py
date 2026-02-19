import glob
import torch
import random

import hydra
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import logging
from source.model import GravNetModel
from source.data import NeutrinoRootDataset


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_files,
        batch_size=16,
        num_workers=4,
        train_test_val_split=(0.7, 0.2, 0.1),
    ):
        super().__init__()
        self.root_files = root_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_test_val_split[0]
        self.test_split = train_test_val_split[1]
        self.val_split = train_test_val_split[2]

        assert abs(self.train_split + self.test_split + self.val_split - 1.0) < 1e-6, "Splits must sum to 1.0"

        self.setup()

    def setup(self, stage=None):
        files = list(self.root_files)

        rng = random.Random(42)
        rng.shuffle(files)

        n = len(files)
        n_test = int(0.2 * n)
        n_val = int(0.1 * n)

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        self.train_ds = NeutrinoRootDataset(train_files)
        self.val_ds   = NeutrinoRootDataset(val_files)
        self.test_ds  = NeutrinoRootDataset(test_files)

        logging.info(f"Train events: {len(self.train_ds)}")
        logging.info(f"Val events:   {len(self.val_ds)}")
        logging.info(f"Test events:  {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    

class GravNetLightning(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        lambda_reg=0.5,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = GravNetModel()

        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.HuberLoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        class_logits, energy_pred = self(batch)

        cls_loss = self.cls_loss(class_logits, batch.y_class)
        reg_loss = self.reg_loss(energy_pred, batch.y_energy)

        loss = cls_loss + self.hparams.lambda_reg * reg_loss

        preds = torch.argmax(class_logits, dim=1)
        acc = (preds == batch.y_class).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_cls_loss", cls_loss)
        self.log("train_reg_loss", reg_loss)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        class_logits, energy_pred = self(batch)

        cls_loss = self.cls_loss(class_logits, batch.y_class)
        reg_loss = self.reg_loss(energy_pred, batch.y_energy)

        loss = cls_loss + self.hparams.lambda_reg * reg_loss

        preds = torch.argmax(class_logits, dim=1)
        acc = (preds == batch.y_class).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


