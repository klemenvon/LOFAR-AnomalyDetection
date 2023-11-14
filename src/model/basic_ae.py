import logging
from typing import Dict
import torch
from torch import nn

from .base_module import TorchModule

log = logging.getLogger(__name__)

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
            activation,
        )
        self.bottleneck_out = nn.Sequential(
            nn.Linear(latent, out_units),
            activation,
        )

    def forward(self,x, **kwargs):
        # Returns (latent, bottleneck_out)
        # Loss dict remains unchanged because we don't have any losses here
        z = self.bottleneck_in(x)
        x_hat = self.bottleneck_out(z)
        return z, x_hat


class Conv2DAutoEncoder(TorchModule):
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

    def forward(self,input):
        x = self.encoder(input)
        x_flat = torch.flatten(x,start_dim=1)
        latent, z = self.bottleneck(x_flat)
        z_shaped = torch.reshape(z,(-1,*self.decoder.input_shape))
        x_hat = self.decoder(z_shaped)
        return input, latent, x_hat

    def loss_function(self,*args,**kwargs):
        # returns the appropriate loss that we need to do backprop
        input, latent, x_hat = args
        mse_loss = nn.MSELoss()
        loss_dict = {}
        loss_dict['mse_loss'] = mse_loss(input,x_hat)
        loss_dict['loss'] = self._scale_loss(loss_dict)
        return loss_dict
  
