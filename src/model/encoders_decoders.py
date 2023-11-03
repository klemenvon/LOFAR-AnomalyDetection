from functools import reduce
from typing import List, Tuple
import torch
from torch import nn

### NOTE: All of this is built assuming an input size of 128x128
# Encoders should return only the last layer of activation
# Decoders should only take the first layer of activation
class EncoderShallow(nn.Module):
    def __init__(self,filters: List[int], input_dim: int = 4, activation: nn.Module = nn.ReLU()):
        """
        A Shallow 2D Convolutional Encoder. Kernel size 7 in first layer, 5 after and 3 in the rest.
        :param filters: filters per layer for remaining layers
        :param input_dim: input dimension
        :param activation: layer activation
        """
        super().__init__()
        kernels = [7,5,3,3,3]
        assert len(filters) == len(kernels), "Number of filters and kernels in Encoder must be equal"

        enc = []
        # Construct encoder by layer blocks
        for fil,kern in zip(filters,kernels):
            layer_list = []
            layer_list.append(nn.Conv2d(input_dim, fil, kernel_size=kern,padding=kern//2))
            layer_list.append(nn.BatchNorm2d(fil))
            layer_list.append(activation)
            layer_list.append(nn.MaxPool2d(2))
            # Add to encoder
            enc.append(nn.Sequential(*layer_list))
        
        self.encoder = nn.Sequential(*enc)
        self.output_shape = (16,16,filters[-1])
        self.output_units = reduce(lambda x,y: x*y, self.output_shape)

    def forward(self,x):
        return self.encoder(x)

class DecoderShallow(nn.Module):
    def __init__(self,filters: List[int], output_dim: int = 4, activation: nn.Module = nn.ReLU(), last_activation: nn.Module = nn.Sigmoid()):
        """
        A Shallow 2D Convolutional Decoder. Kernel size 3 in first layer, 5 after and 7 in the rest.
        :param filters: filters per layer for remaining layers
        :param output_dim: output dimension
        :param activation: layer activation
        :param last_activation: activation of last layer
        """
        super().__init__()
        kernels = [3,5,7,7,7]
        assert len(filters) == len(kernels), "Number of filters and kernels in Decoder must be equal"

        dec = []
        # Construct decoder by layer blocks
        for i in range(len(filters) - 1):
            layer_list = []
            layer_list.append(nn.ConvTranspose2d(filters[i], filters[i+1], kernel_size=kernels[i],padding=kernels[i]//2))
            layer_list.append(nn.BatchNorm2d(filters[i+1]))
            layer_list.append(activation)
            layer_list.append(nn.Upsample(scale_factor=2))
            # Add to decoder
            dec.append(nn.Sequential(*layer_list))
        
        # Make the last layer
        layer_list = []
        layer_list.append(nn.ConvTranspose2d(filters[-1], output_dim, kernel_size=kernels[-1],padding=kernels[-1]//2))
        layer_list.append(last_activation)
        layer_list.append(nn.Upsample(scale_factor=2))
        dec.append(nn.Sequential(*layer_list))

        self.decoder = nn.Sequential(*dec)
        self.input_shape = (16,16,filters[0])
        self.input_units = reduce(lambda x,y: x*y, self.input_shape)

    def forward(self,x):
        return self.decoder(x)

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
        :param filters: the number of filters per layer of the decoder :param kernels: the kernel size per layer of the decoder
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
