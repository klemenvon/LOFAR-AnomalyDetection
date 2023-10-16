from typing import Any
import torch
from torch import nn
import pytorch_lightning as pl

class Conv2DAutoEncoder(pl.LightningModule):
    def __init__(self,encoder,decoder,bottleneck):
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    def forward(self,x):
        x = self.encoder(x)
        latent, z = self.bottleneck(x)
        x_hat = self.decoder(z)
        return latent, x_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        latent, x_hat = self(x)
        loss = nn.MSELoss(x_hat,x)
        self.log('train_loss', loss)
        return loss
    

