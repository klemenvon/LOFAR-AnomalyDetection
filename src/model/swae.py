import logging
from typing import Dict

import torch
from torch import nn

from .base_module import TorchModule
from .basic_ae import BasicBottleneck

log = logging.getLogger(__name__)

class SWAE(TorchModule):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            latent: int,
            loss_scaling: Dict[str,float],
            activation: nn.Module = nn.ReLU(),
            wasserstein_deg: int = 2,
            num_projections: int = 50,
            reg_weight: float = 100.0
    ):
        """
        Sliced Wasserstein Auto Encoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = BasicBottleneck(encoder.output_units, decoder.input_units, latent, activation)
        self.loss_scaling = loss_scaling
        self.wasserstein_deg = wasserstein_deg
        self.num_projections = num_projections
        self.reg_weight = reg_weight

    def forward(self,input):
        x = self.encoder(input)
        x_flat = torch.flatten(x,start_dim=1)
        latent, z = self.bottleneck(x_flat)
        z_shaped = torch.reshape(z,(-1,*self.decoder.input_shape))
        x_hat = self.decoder(z_shaped)
        return input, latent, x_hat
    
    def loss_function(self,*args,**kwargs):
        input, latent, x_hat = args
        mse_loss = nn.MSELoss()
        loss_dict = {}
        loss_dict['mse_loss'] = mse_loss(input,x_hat)
        loss_dict['swae_loss'] = self._compute_swd(latent)
        loss_dict['loss'] = self._scale_loss(loss_dict)
        return loss_dict
    
    def get_random_projections(self,latent_dim,num_projections):
        rand_samples = torch.randn(num_projections,latent_dim)
        rand_proj = rand_samples /rand_samples.norm(dim=1).view(-1,1)
        return rand_proj

    def _compute_swd(self,latent):
        device = latent.device
        batch, latent_dim = latent.shape
        prior = torch.randn_like(latent).to(device,non_blocking=True)
        projection_matrix = self.get_random_projections(latent_dim,self.num_projections).transpose(0,1).to(device,non_blocking=True)
        latent_projections = latent.matmul(projection_matrix)
        prior_projections = prior.matmul(projection_matrix)
        w_dist = torch.sort(latent_projections.t(),dim=1)[0] - \
                torch.sort(prior_projections.t(),dim=1)[0]
        w_dist = w_dist.pow(self.wasserstein_deg)
        return self.reg_weight * w_dist.mean()
