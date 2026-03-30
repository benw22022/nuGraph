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
from source.spconv_model import Sparse3DFlowRegression
import torch.nn as nn 
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from source.test import load_run_config


class GravNetLightning(pl.LightningModule):
    def __init__(
        self,
        model,
        targets=["E_nu"],
        lr=1e-3,
        lambda_reg=0.25,
    ):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.targets = targets
        # self.reg_loss = nn.HuberLoss()
        # self.reg_loss = nn.MSELoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):

        target_truth = {t: batch[t] for t in self.targets if hasattr(batch, t)}
        
        y_true = torch.stack(
        [
            target_truth[t].float()
            for t in self.targets
        ],
        dim=1,  # (batch, n targets)
        )

        loss = self.model.loss(batch, y_true)

        self.log("train_loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        
        target_truth = {t: batch[t] for t in self.targets if hasattr(batch, t)}
        
        y_true = torch.stack(
        [
            target_truth[t].float()
            for t in self.targets
        ],
        dim=1,  # (batch, n targets)
        )

        loss = self.model.loss(batch, y_true)

        self.log("val_loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)

        return loss
        

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



    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), self.lr)

    #     warmup = LinearLR(
    #         optimizer,
    #         start_factor=self.lr * 0.001,
    #         end_factor=1.0,
    #         total_iters=1000
    #     )

    #     cosine = CosineAnnealingLR(optimizer, T_max=20000)

    #     scheduler = SequentialLR(
    #         optimizer,
    #         schedulers=[warmup, cosine],
    #         milestones=[1000]
    #     )

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #         },
    #     }
    
    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)


def run_spconv_flow_training(cfg):
    
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
        model=Sparse3DFlowRegression(cfg.model).to(device),
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
    
    # trainer.test(model, datamodule=datamodule)

    # model = GravNetLightning(
    #     model=FastGravNet(3),
    #     lr=1e-3,
    # )

    # # logger = TensorBoardLogger()
    #         # save_dir=cfg.logging.log_dir,
    #         # save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,    )

    # pt_files = glob.glob("data/pt/*/*.pt")
    # assert len(pt_files) > 0, "No .pt files found" 

    # datamodule = GraphDataModule(
    #     pt_files=pt_files,
    #     batch_size=8,
    #     num_workers=16,
    # )


    # checkpoint_cb = ModelCheckpoint(
    #     monitor="val_loss",
    #     save_top_k=1,
    #     mode="min",
    # )  

    # early_stop_cb = EarlyStopping(
    # monitor="val_loss",
    # min_delta=0,
    # patience=12,
    # mode="min",
    # verbose=True,
    # )
    
    # # datamodule.setup()
    # # for batch in datamodule.train_dataloader():
    # #     print(batch.y_class.min(), batch.y_class.max())
    #     # break


    # trainer = Trainer(
    #     max_epochs=1000,
    #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #     devices=1,
    #     callbacks=[checkpoint_cb, early_stop_cb],
    #     log_every_n_steps=50,
    #     accumulate_grad_batches=8,
    #     benchmark=True,
    #     precision="16-mixed",
    # )

    # trainer.fit(model, datamodule=datamodule)