from typing import Dict
import torch
from torch import nn

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
            activation,
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,x):
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
        x_flat = torch.flatten(x,start_dim=1)
        z, mu, logvar = self.bottleneck(x_flat)
        z_shaped = torch.reshape(z,(-1,*self.decoder.input_shape))
        x_hat = self.decoder(z_shaped)
        return input, mu, logvar, x_hat
    
    def _kl_divergence(self,mu,logvar):
        kl_divergence = torch.mean(-0.5 * torch.sum(1+logvar - mu ** 2 - logvar.exp(),dim=1),dim=0)
        return kl_divergence

    def loss_function(self,*args,**kwargs):
        # returns the appropriate loss that we need to do backprop
        input, mu, logvar, x_hat = args
        mse_loss = nn.MSELoss()
        loss_dict = {}
        loss_dict['mse_loss'] = mse_loss(input,x_hat)
        loss_dict['kld_loss'] = self._kl_divergence(mu,logvar)
        loss_dict['loss'] = self._scale_loss(loss_dict)
        return loss_dict
  

