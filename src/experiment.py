# The base file for the Pytorch Lightning module
# Some structure borrowed from https://github.com/AntixK/PyTorch-VAE
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

class ExperimentBase(pl.LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        super(ExperimentBase, self).__init__()
        # Initialize model and set arguments
        self.model = model
        self.config = config

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        # Get result tuple
        result = self.forward(input)
        # Get the training loss dictionary
        training_loss = self.model.loss_function(
            *result,
            batch_idx = batch_idx
        )
        self.log_dict({key: val.item() for key,val in training_loss.items()})
        return training_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(
            *results,
            batch_idx = batch_idx
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        # Get optimizer configuration from config
        optimizer_config = self.config.optimizer
        name = optimizer_conig.pop('name')
        if name == "adam":
            optimizer = optim.Adam(self.model.parameters(), **optimizer_config)
        elif name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), **optimizer_config)
        else:
            log.error(f"Unknown Optimizer {name}. Failed to set up optimizer.")
        return optimizer


