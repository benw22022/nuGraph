import glob
import torch
import random
import os
import hydra
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import logging

from source.dataset import GraphDataset, CombinedDataset, GraphDataModule
from source.diffusion_model import DiffusionSchedule, SparseDiffusionModel, q_sample
import torch.nn as nn 
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from source.test import load_run_config
import spconv.pytorch as spconv

class GravNetLightning(pl.LightningModule):
    def __init__(
        self,
        model,
        targets=["E_nu"],
        lr=1e-3,
        lambda_reg=0.25,
        T=1000,
        stats=None,
    ):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.targets = targets
        self.schedule = DiffusionSchedule(T=T)

        # self.reg_loss = nn.HuberLoss()
        # self.reg_loss = nn.MSELoss()

    def forward(self, sparse_tensor, y_t, t):
        return self.model(sparse_tensor, y_t, t)

    
    def setup(self, stage=None):
        # move schedule to correct device
        self.schedule = self.schedule.to(self.device)

    def _prepare_batch(self, batch):
        """
        Convert your PyG-like batch into:
        - SparseConvTensor
        - target tensor y
        """

        coords = batch.x.int()   # [N, 3] → (x,y,z)
        batch_idx = batch.batch.unsqueeze(1)

        # spconv expects (batch, z, y, x)
        indices = torch.cat([batch_idx, coords[:, [2,1,0]]], dim=1)
        indices = indices.to(torch.int32)

        features = torch.ones(coords.shape[0], 1, device=self.device)

        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[128,128,128],  # adjust!
            # spatial_shape=[2048, 2048, 2048],  # adjust!
            batch_size=batch.num_graphs,
        )

        target_truth = {t: batch[t] for t in self.targets if hasattr(batch, t)}
        y_true = torch.stack(
            [
                target_truth[t].float()
                for t in self.targets
            ],
            dim=1,  # (batch, n targets)
            )
        return sparse_tensor, y_true
    

    def training_step(self, batch, batch_idx):
        sparse_tensor, y0 = self._prepare_batch(batch)

        B = y0.size(0)
        device = y0.device

        t = torch.randint(0, self.schedule.T, (B,), device=device)

        y_t, noise = q_sample(y0, t, self.schedule)

        noise_pred = self(sparse_tensor, y_t, t)

        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        sparse_tensor, y0 = self._prepare_batch(batch)

        B = y0.size(0)
        device = y0.device

        t = torch.randint(0, self.schedule.T, (B,), device=device)

        y_t, noise = q_sample(y0, t, self.schedule)
        noise_pred = self(sparse_tensor, y_t, t)

        loss = F.mse_loss(noise_pred, noise)

        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    @torch.no_grad()
    def sample(self, batch):
        sparse_tensor, _ = self._prepare_batch(batch)

        context = self.model.encoder(sparse_tensor)

        B = context.size(0)
        y_dim = len(self.targets)
        device = context.device

        y_t = torch.randn(B, y_dim, device=device)

        for t in reversed(range(self.schedule.T)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            noise_pred = self.model.diffusion(y_t, t_tensor, context)

            alpha = self.schedule.alpha[t]
            alpha_bar = self.schedule.alpha_bar[t]
            beta = self.schedule.beta[t]

            noise = torch.randn_like(y_t) if t > 0 else 0

            y_t = (
                (1 / torch.sqrt(alpha)) *
                (y_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred)
                + torch.sqrt(beta) * noise
            )

        return y_t


def run_diffusion_training(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logger = TensorBoardLogger(
        # save_dir=cfg.logging.log_dir,
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    
    pt_files = []
    for run in cfg.data.runs:
        run_files = glob.glob(os.path.join(cfg.data.datapath, str(run), "*.pt"))
        logging.info(f"Found {len(run_files)} for run {run}")
        pt_files += run_files
    assert len(pt_files) > 0, "No .pt files found" 
    logging.info(f"Found {len(pt_files)} files total")

    datamodule = GraphDataModule(
        pt_files=pt_files,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    model = GravNetLightning(
        model=SparseDiffusionModel(cfg.model.n_targets, input_channels=1).to(device),
        targets=cfg.training.targets,
        lr=cfg.training.learning_rate,
        lambda_reg=cfg.training.regression_loss_scale,
    )

    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )  
    
    early_stop_cb = EarlyStopping(
    monitor=cfg.training.early_stopping.monitor,
    min_delta=cfg.training.early_stopping.min_delta,
    patience=cfg.training.early_stopping.patience,
    mode="min",
    verbose=True,
    )


    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=5,
        # precision="16-mixed",
        # accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        # gradient_clip_val=1.0
    )

    trainer.fit(model, datamodule=datamodule)
    
