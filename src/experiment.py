# The base file for the Pytorch Lightning module
import torch
from torch import nn
import pytorch_lightning as pl

class ExperimentBase(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            

    )