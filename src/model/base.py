from functools import reduce
from typing import List, Tuple
import torch
from torch import nn


class Conv2DEncoderShallow(nn.Module):
    def __init__(
            self,
            input_dim: int = 4,
            filters: List[int] = [32, 64, 128, 256, 512],
            kernels: List[int] = [7, 5, 3, 3, 3],
            last_shape: Tuple[int,int] = (16,16),
            downsampling: str = 'maxpool',
            activation: nn.Module = nn.ReLU()
    ):
        """
        A 2D Convolutional Encoder with a shallow architecture
        :param input_dim: input dimension
        :param filters: filters per layer for remaining layers
        :param kernels: kernel size per layer
        :param last_shape: shape of the last layer
        :param downsampling: downsampling strategy
        :param activation: layer activation
        """
        super().__init__()
        layer_in = input_dim
        stride = 2 if downsampling == 'stride' else 1

        # Construct the encoder
        enc = []
        for fil,kern in zip(filters,kernels):
            layer_list = []
            layer_list.append(nn.Conv2d(layer_in, fil, kernel_size=kern,padding=kern//2,stride=stride))
            layer_list.append(nn.BatchNorm2d())
            layer_list.append(activation)
            if downsampling == 'maxpool':
                layer_list.append(nn.MaxPool2d(2))
            enc.append(nn.Sequential(*layer_list))
        
        self.encoder = nn.Sequential(*enc)

        # Calculate the output shape
        self.last_shape = last_shape
        self.output_shape = reduce(lambda x,y: x*y, last_shape)
    
    def forward(self, x):
        return nn.Flatten(self.encoder(x))

class Conv2DDecoderShallow(nn.Module):
    def __init__(
            self,
            output_dim: int = 4,
            filters: List[int] = [512, 256, 128, 64, 32],
            kernels: List[int] = [3, 3, 3, 5, 7],
            first_shape: Tuple[int,int] = (16,16),
            upsampling: str = 'upsample',
            activation: nn.Module = nn.ReLU(),
            last_activation: nn.Module = nn.Sigmoid()
    ):
        """
        A 2D Convolutional Decoder witih a shallow architecture
        :param output_dim: the dimension of the output
        :param filters: the number of filters per layer of the decoder
        :param kernels: the kernel size per layer of the decoder
        :param first_shape: the shape of the first layer
        :param upsampling: the upsampling strategy
        :param activation: the layer activation
        :param last_activation: the layer activation of the last layer
        """
        super().__init__()
        filters += [output_dim]
        stride = 2 if upsampling == 'stride' else 1
        out_padding = 1 if upsampling == 'stride' else 0

        dec = []
        for i in range(len(kernels) - 1):
            layer_list = []
            layer_list.append(nn.ConvTranspose2d(filters[i], filters[i+1], kernel_size=kernels[i],padding=kernels[i]//2,stride=stride,output_padding=out_padding))
            layer_list.append(nn.BatchNorm2d())
            layer_list.append(activation)
            if upsampling == 'upsample':
                layer_list.append(nn.Upsample(scale_factor=2))
            dec.append(nn.Sequential(*layer_list))
        
        # Make the last layer
        layer_list = []
        layer_list.append(nn.ConvTranspose2d(filters[-2], filters[-1], kernel_size=kernels[-1],padding=kernels[-1]//2,stride=stride,output_padding=out_padding))
        layer_list.append(last_activation)
        if upsampling == 'upsample':
            layer_list.append(nn.Upsample(scale_factor=2))
        dec.append(nn.Sequential(*layer_list))

        self.decoder = nn.Sequential(*dec)

        # Calculate the input shape flattened (a bit unintuitive but we reshape it in the forward pass)
        self.input_shape = reduce(lambda x,y: x*y, first_shape)

    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        return self.decoder(x)
