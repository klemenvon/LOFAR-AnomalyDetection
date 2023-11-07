from typing import Dict
import torch
from torch import nn
from torch.nn.functional import MSELoss

from .base_module import TorchModule

class VAEBottleneck(nn.Module):
    def __init__(self, in_units: int, out_units: int, latent: int, activation: nn.Module = nn.ReLU()):
        """
        A basic bottleneck layer for a convolutional autoencoder.
        :param in_shape: input shape
        :param out_shape: output shape
        """
        super().__init__()
        # Print the arguments
        self.fc_var = nn.Linear(in_units, latent)
        self.fc_mu = nn.Linear(in_units, latent)
        self.bottleneck_out = nn.Sequential(
            nn.Linear(latent, out_units),
            nn.BatchNorm2d(out_units),
            activation,
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,x,loss_dict):
        # Returns (latent, bottleneck_out)
        # Loss dict remains unchanged because we don't have any losses here
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.bottleneck_out(z)
        return x_hat, mu, logvar


class VAE(TorchModule):
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
        self.bottleneck = VAEBottleneck(encoder.output_units, decoder.input_units, latent, activation)
        self.loss_scaling = loss_scaling

    def forward(self,input):
        x = self.encoder(input)
        z, mu, logvar = self.bottleneck(x)
        x_hat = self.decoder(z)
        return input, mu, logvar, x_hat

    def loss_function(self,*args,**kwargs):
        # returns the appropriate loss that we need to do backprop
        input, latent, x_hat = args
        mse_loss = MSELoss()
        loss_dict = {}
        loss_dict['mse_loss'] = mse_loss(input,x_hat)
        loss_dict['loss'] = self._scale_loss(loss_dict)
        return loss_dict
  

