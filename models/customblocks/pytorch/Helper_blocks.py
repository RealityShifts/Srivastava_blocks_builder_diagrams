from __future__ import annotations



import torch
import torch.nn as nn
from jaxtyping import Float, Shaped
from torch import Tensor
from pytorch_blocks.core_blocks import AdaIN, ConvBlock, ResidualBlock
from pytorch_blocks.attention_blocks import MultiHeadAttention
from pytorch_blocks.unet_diffusion_blocks import DownsampleBlock
from einops import rearrange


class ResidualChain(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,repeats:int =3):
        super().__init__()
        self.repeats = repeats
        self.inital_conv = ResidualBlock(in_channels,out_channels,norm = "instance")
        self.res_chain = nn.ModuleList([ResidualBlock(out_channels ,out_channels, norm = "instance") for _ in range(self.repeats)])
        
    
    def forward(self,x:Float[Tensor,"B C_in H W"])->Float[Tensor,"B C_out H W"]:
        x = self.inital_conv(x)
        for block in self.res_chain:
            x = block(x)
        return x
    

