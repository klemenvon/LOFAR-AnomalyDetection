from typing import Dict
import torch
from torch import nn
import pytorch_lightning as pl

class BasicBottleneck(nn.Module):
    def __init__(self, in_units: int, out_units: int, latent: int, activation: nn.Module = nn.ReLU()):
        """
        A basic bottleneck layer for a convolutional autoencoder.
        :param in_shape: input shape
        :param out_shape: output shape
        """
        super().__init__()
        # Print the arguments
        self.bottleneck_in = nn.Sequential(
            nn.Linear(in_units, latent),
            nn.BatchNorm2d(latent),
            activation,
        )
        self.bottleneck_out = nn.Sequential(
            nn.Linear(latent, out_units),
            nn.BatchNorm2d(out_units),
            activation,
        )

    def forward(self,x,loss_dict):
        # Returns (latent, bottleneck_out)
        # Loss dict remains unchanged because we don't have any losses here
        z = self.bottleneck_in(x)
        x_hat = self.bottleneck_out(z)
        return z, x_hat

class Conv2DAutoEncoder(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            latent: int,
            loss_scaling: Dict[str,float],
            activation: nn.Module = nn.ReLU()
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = BasicBottleneck(encoder.output_units, decoder.input_units, latent, activation)
        self.loss_scaling = loss_scaling
    
    def _log_losses(self,loss_dict):
        for key in loss_dict:
            self.log(key, loss_dict[key])
    
    def _scale_loss(self,loss_dict):
        # Scale the losses and add them up
        loss = None
        for key in loss_dict:
            current = loss_dict[key] * self.loss_scaling[key]
            if loss is None:
                loss = current
            else:
                loss += current
        return loss

    def forward(self,x,loss_dict):
        x = self.encoder(x)
        latent, z = self.bottleneck(x)
        x_hat = self.decoder(z)
        return latent, x_hat
    
    def training_step(self, batch, batch_idx):
        loss_dict = {}
        x = self.encoder(batch)
        latent, z = self.bottleneck(x,loss_dict)
        x_hat = self.decoder(z)
        loss_dict['mse_loss'] = nn.MSELoss(x_hat,batch)
        loss = self._scale_loss(loss_dict)
        self._log_losses(loss_dict)
        self.log('train_loss', loss)
        return loss

    