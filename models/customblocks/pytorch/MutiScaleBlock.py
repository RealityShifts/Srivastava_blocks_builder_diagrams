from __future__ import annotations



import torch
import torch.nn as nn
from jaxtyping import Float, Shaped
from torch import Tensor
from pytorch_blocks.core_blocks import AdaIN, ConvBlock, ResidualBlock
from pytorch_blocks.attention_blocks import MultiHeadAttention
from pytorch_blocks.unet_diffusion_blocks import DownsampleBlock
from einops import rearrange

class Face_to_embedding(nn.Module):
    def __init__(self, norm: str = "layer"):
        super().__init__()
        self._9vbcl = ConvBlock(in_ch=3, out_ch=32, norm=norm)
        self.wahuw = ConvBlock(in_ch=32, out_ch=32, norm=norm)
        self.m13dc = ConvBlock(in_ch=32, out_ch=32, norm=norm)
        self._71pl7 = ConvBlock(in_ch=32, out_ch=64, norm=norm)
        self.tt1yo = ConvBlock(in_ch=64, out_ch=128, norm=norm)
        self._3qwwo = ConvBlock(in_ch=128, out_ch=128, norm=norm)
        self.z4ujm = ConvBlock(in_ch=128, out_ch=128, norm=norm)
        self._90h0t = ConvBlock(in_ch=128, out_ch=128, norm=norm)
        self.cw2et = ConvBlock(in_ch=128, out_ch=128, norm=norm)
        self.ye3dv = ConvBlock(in_ch=128, out_ch=128, norm=norm)
        self.bxnf2 = ConvBlock(in_ch=128, out_ch=128, norm=norm)
        self.tja0y = ConvBlock(in_ch=128, out_ch=128, norm=norm)

    def forward(self, in0: Shaped[Tensor, "B C_in H W"]) -> Shaped[Tensor, "b _ c"]:
        _9vbcl = self._9vbcl(x=in0)
        wahuw = self.wahuw(x=_9vbcl)
        m13dc = self.m13dc(x=wahuw)
        e17jo = torch.nn.functional.max_pool2d(m13dc, kernel_size=2, stride=2, padding=0)
        _71pl7 = self._71pl7(x=e17jo)
        tt1yo = self.tt1yo(x=_71pl7)
        _3qwwo = self._3qwwo(x=tt1yo)
        _0kelz = torch.nn.functional.max_pool2d(_3qwwo, kernel_size=2, stride=2, padding=0)
        z4ujm = self.z4ujm(x=_0kelz)
        _90h0t = self._90h0t(x=z4ujm)
        cw2et = self.cw2et(x=_90h0t)
        xtp2h = torch.nn.functional.max_pool2d(cw2et, kernel_size=2, stride=2, padding=0)
        ye3dv = self.ye3dv(x=xtp2h)
        bxnf2 = self.bxnf2(x=ye3dv)
        tja0y = self.tja0y(x=bxnf2)
        xtp2h2 = torch.nn.functional.max_pool2d(tja0y, kernel_size=2, stride=2, padding=0)
        ye3dv2 = self.ye3dv(x=xtp2h2)
        bxnf22 = self.bxnf2(x=ye3dv2)
        tja0y2 = self.tja0y(x=bxnf22)
        _80df9 = rearrange(tja0y2, "b c h w -> b (h w) c")
        out0 = _80df9
        return out0


class Self_attention_layerrs(nn.Module):
    def __init__(self):
        super().__init__()
        self._4qiau = MultiHeadAttention(dim=128)
        self.w9q1b = ResidualBlock(in_ch=128, out_ch=128)
        self._1wuei = MultiHeadAttention(dim=128)
        self.f4o5q = ResidualBlock(in_ch=128, out_ch=128)
        self.rb2lm = MultiHeadAttention(dim=128)
        self._04ew7 = ResidualBlock(in_ch=128, out_ch=128)

    def forward(self, in0: Shaped[Tensor, "B Tq D"], in1: Shaped[Tensor, "B Tk D"] = None, in2: Shaped[Tensor, "B Tk D"] = None, in3: Shaped[Tensor, "..."] = None, in4: Shaped[Tensor, "..."] = None, in5: Shaped[Tensor, "..."] = None) -> Shaped[Tensor, "b _ c"]:
        _4qiau = self._4qiau(query=in0, key=in1, value=in2, mask=in3)
        rearranger = rearrange(_4qiau, "b (h w) c -> b c h w", h=12, w=12)
        w9q1b = self.w9q1b(x=rearranger)
        jp5nl = rearrange(w9q1b, "b c h w -> b (h w) c")
        _1wuei = self._1wuei(query=jp5nl, key=jp5nl, value=jp5nl, mask=in4)
        rearranger2 = rearrange(_1wuei, "b (h w) c -> b c h w", h=12, w=12)
        f4o5q = self.f4o5q(x=rearranger2)
        fxf9q = rearrange(f4o5q, "b c h w -> b (h w) c")
        rb2lm = self.rb2lm(query=fxf9q, key=fxf9q, value=fxf9q, mask=in5)
        rearranger3 = rearrange(rb2lm, "b (h w) c -> b c h w", h=12, w=12)
        _04ew7 = self._04ew7(x=rearranger3)
        fxf9q2 = rearrange(_04ew7, "b c h w -> b (h w) c")
        out0 = fxf9q2
        return out0


class Decoder(nn.Module):
    def __init__(self, norm: str = "layer", style_dim: int = 256):
        super().__init__()
        self._1pefw = ResidualBlock(in_ch=128, out_ch=128, norm=norm)
        self.o3qgz = ResidualBlock(in_ch=128, out_ch=128, norm=norm)
        self.ivgok = ConvBlock(in_ch=128, out_ch=64, norm=norm)
        self.y5uut = AdaIN(num_features=64, style_dim=style_dim)
        self.cz8bw = ResidualBlock(in_ch=64, out_ch=64, norm=norm)
        self.kz2zw = ResidualBlock(in_ch=64, out_ch=64, norm=norm)
        self._7j0vj = ConvBlock(in_ch=64, out_ch=32, norm=norm)
        self._83c8m = AdaIN(num_features=32, style_dim=style_dim)
        self._3ypq8 = ResidualBlock(in_ch=32, out_ch=64, norm=norm)
        self.ouqy9 = ResidualBlock(in_ch=64, out_ch=32, norm=norm)
        self.cyp8k = ConvBlock(in_ch=32, out_ch=16, norm=norm)
        self.jzynm = AdaIN(num_features=16, style_dim=style_dim)
        self.h6vw0 = ResidualBlock(in_ch=16, out_ch=16, norm=norm)
        self.m1cep = ResidualBlock(in_ch=16, out_ch=16, norm=norm)
        self._1mrh4 = ResidualBlock(in_ch=16, out_ch=16, norm="none")
        self._3g5xd = ConvBlock(in_ch=16, out_ch=3, norm="none")

    def forward(self, in0: Shaped[Tensor, "B C_in H W"], in1: Shaped[Tensor, "B D_style"], in2: Shaped[Tensor, "B D_style"], in3: Shaped[Tensor, "B D_style"]) -> Shaped[Tensor, "B C_out H_out W_out"]:
        _1pefw = self._1pefw(x=in0)
        o3qgz = self.o3qgz(x=_1pefw)
        avsi5 = torch.nn.functional.interpolate(o3qgz, scale_factor=2, mode='bilinear', align_corners=False)
        ivgok = self.ivgok(x=avsi5)
        y5uut = self.y5uut(x=ivgok, style=in1)
        cz8bw = self.cz8bw(x=y5uut)
        kz2zw = self.kz2zw(x=cz8bw)
        s7sbo = torch.nn.functional.interpolate(kz2zw, scale_factor=2, mode='bilinear', align_corners=False)
        _7j0vj = self._7j0vj(x=s7sbo)
        _83c8m = self._83c8m(x=_7j0vj, style=in2)
        _3ypq8 = self._3ypq8(x=_83c8m)
        ouqy9 = self.ouqy9(x=_3ypq8)
        hn6sg = torch.nn.functional.interpolate(ouqy9, scale_factor=2, mode='bilinear', align_corners=False)
        cyp8k = self.cyp8k(x=hn6sg)
        jzynm = self.jzynm(x=cyp8k, style=in3)
        h6vw0 = self.h6vw0(x=jzynm)
        m1cep = self.m1cep(x=h6vw0)
        _34rq7 = torch.nn.functional.interpolate(m1cep, scale_factor=2, mode='bilinear', align_corners=False)
        _1mrh4 = self._1mrh4(x=_34rq7)
        _3g5xd = self._3g5xd(x=_1mrh4)
        out0 = _3g5xd
        return out0


class Style(nn.Module):
    def __init__(self, channels: int = 256, in_ch: int = 256, out_ch: int = 256):
        super().__init__()
        self.gkm86 = ConvBlock(in_ch=128, out_ch=channels)
        self.pe5r3 = DownsampleBlock(channels=channels)
        self.dhzry = ConvBlock(in_ch=in_ch, out_ch=out_ch)
        self.wlbkp = DownsampleBlock(channels=channels)
        self.bnogs = ConvBlock(in_ch=channels, out_ch=256, stride=3)

    def forward(self, in0: Shaped[Tensor, "B C_in H W"]) -> Shaped[Tensor, "b _"]:
        gkm86 = self.gkm86(x=in0)
        pe5r3 = self.pe5r3(x=gkm86)
        dhzry = self.dhzry(x=pe5r3)
        wlbkp = self.wlbkp(x=dhzry)
        bnogs = self.bnogs(x=wlbkp)
        wmjhf = rearrange(bnogs, "b c h w -> b (h w c) ")
        out0 = wmjhf
        return out0


class Upscaler(nn.Module):
    """Upscaler model.

Architecture overview:

```mermaid
flowchart TD
    n_8ffdcc56520da835(["Image_face"])
    n_c899b9bcd4005339(["Image_bg"])
    n_54893d9d334bfe50["MultiHeadAttention"]
    n_ab74635a522acc63["Rearrange"]
    n_78ba337f89ab3f5c(["y"])
    n_7a073cfc035d1bce[["Face_to_embedding"]]
    n_db8dd9d6a4346c08[["Face_to_embedding"]]
    n_a3488d9be4648906[["Face_to_embedding · bg_ext"]]
    n_614b63cea3f0b4a7[["self_attention_layerrs · facial_self_attn"]]
    n_256ebd706e8d3061[["self_attention_layerrs · facial_self_attn"]]
    n_4ef73d42dbaa1b19[["self_attention_layerrs · body_self_attn"]]
    n_3829ab89cec63a6f[["decoder · hmenl"]]
    n_e440b70dd2b62cdf[["Style · gwrhh"]]
    n_2e44d2f04f2662c8[["Style · mxm33"]]
    n_54893d9d334bfe50 --> n_ab74635a522acc63
    n_8ffdcc56520da835 --> n_7a073cfc035d1bce
    n_c899b9bcd4005339 --> n_db8dd9d6a4346c08
    n_c899b9bcd4005339 --> n_a3488d9be4648906
    n_614b63cea3f0b4a7 --> n_54893d9d334bfe50
    n_7a073cfc035d1bce --> n_614b63cea3f0b4a7
    n_7a073cfc035d1bce --> n_614b63cea3f0b4a7
    n_7a073cfc035d1bce --> n_614b63cea3f0b4a7
    n_256ebd706e8d3061 --> n_54893d9d334bfe50
    n_db8dd9d6a4346c08 --> n_256ebd706e8d3061
    n_db8dd9d6a4346c08 --> n_256ebd706e8d3061
    n_db8dd9d6a4346c08 --> n_256ebd706e8d3061
    n_4ef73d42dbaa1b19 --> n_54893d9d334bfe50
    n_a3488d9be4648906 --> n_4ef73d42dbaa1b19
    n_a3488d9be4648906 --> n_4ef73d42dbaa1b19
    n_a3488d9be4648906 --> n_4ef73d42dbaa1b19
    n_e440b70dd2b62cdf --> n_3829ab89cec63a6f
    n_e440b70dd2b62cdf --> n_3829ab89cec63a6f
    n_2e44d2f04f2662c8 --> n_3829ab89cec63a6f
    n_ab74635a522acc63 --> n_3829ab89cec63a6f
    n_3829ab89cec63a6f --> n_78ba337f89ab3f5c
    n_ab74635a522acc63 --> n_e440b70dd2b62cdf
    n_ab74635a522acc63 --> n_2e44d2f04f2662c8
    style n_8ffdcc56520da835 fill:#dd4b4b,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_c899b9bcd4005339 fill:#dd4bb8,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_78ba337f89ab3f5c fill:#e7665f,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_7a073cfc035d1bce fill:#b465e2,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_db8dd9d6a4346c08 fill:#b465e2,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_a3488d9be4648906 fill:#b465e2,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_614b63cea3f0b4a7 fill:#45e3b0,stroke:#0b0e16,stroke-width:1px,color:#10131f
    style n_256ebd706e8d3061 fill:#45e3b0,stroke:#0b0e16,stroke-width:1px,color:#10131f
    style n_4ef73d42dbaa1b19 fill:#45e3b0,stroke:#0b0e16,stroke-width:1px,color:#10131f
    style n_3829ab89cec63a6f fill:#e85aed,stroke:#0b0e16,stroke-width:1px,color:#f5f6fa
    style n_e440b70dd2b62cdf fill:#b7dd4b,stroke:#0b0e16,stroke-width:1px,color:#10131f
    style n_2e44d2f04f2662c8 fill:#b7dd4b,stroke:#0b0e16,stroke-width:1px,color:#10131f
```
"""
    def __init__(self):
        super().__init__()
        self.Face_to_embedding = Face_to_embedding()
        self.bg_ext = Face_to_embedding()
        self.facial_self_attn = Self_attention_layerrs()
        self.body_self_attn = Self_attention_layerrs()
        self.multi_head_attention_8 = MultiHeadAttention(dim=128)
        self.gwrhh = Style()
        self.mxm33 = Style()
        self.hmenl = Decoder()

    def forward(self, Image_face: Float[Tensor, "7 3 192 192"], Image_bg: Float[Tensor, "7 3 192 192"]) -> Shaped[Tensor, "B C_out H_out W_out"]:
        Face_to_embedding = self.Face_to_embedding(in0=Image_face)
        Face_to_embedding2 = self.Face_to_embedding(in0=Image_bg)
        bg_ext = self.bg_ext(in0=Image_bg)
        facial_self_attn = self.facial_self_attn(in0=Face_to_embedding, in1=Face_to_embedding, in2=Face_to_embedding)
        facial_self_attn2 = self.facial_self_attn(in0=Face_to_embedding2, in1=Face_to_embedding2, in2=Face_to_embedding2)
        body_self_attn = self.body_self_attn(in0=bg_ext, in1=bg_ext, in2=bg_ext)
        multi_head_attention_8 = self.multi_head_attention_8(query=facial_self_attn, key=facial_self_attn2, value=body_self_attn)
        rearrange_9 = rearrange(multi_head_attention_8, "b (h w) c -> b c h w", h=12, w=12)
        gwrhh = self.gwrhh(in0=rearrange_9)
        mxm33 = self.mxm33(in0=rearrange_9)
        hmenl = self.hmenl(in0=rearrange_9, in1=gwrhh, in2=gwrhh, in3=mxm33)
        backgroundandFace = hmenl
        return backgroundandFace
