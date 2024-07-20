import re
import math
import torch
from torch import nn
from functools import partial
from einops import rearrange
from timm.layers import LayerNorm2d
from timm.layers.norm_act import LayerNormAct2d
from timm.models.regnet import RegStage
from torchvision.ops.misc import SqueezeExcitation as SELayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SELayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = self.mlp(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class LDPNetProjector(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)


class SPP(nn.Module):

    def __init__(self, config=None, projector_type='v1'):
        super().__init__()

        self.projector_type = projector_type

        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear_0 = nn.Linear(inc, inc)

        self.linear_1 = nn.Linear(inc, ouc)

        self.pooling = nn.AvgPool2d(kernel_size=2)

        self.linear_2 = nn.Linear(ouc, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        if 'v1' in self.projector_type:
            x = self.linear_1(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.linear_2(x)
        elif 'v2' in self.projector_type:
            x = self.linear_1(x)
            x = self.linear_2(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
        elif 'v3' in self.projector_type:
            x = self.linear_0(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.linear_1(x)
            x = self.linear_2(x)
        return x


def build_pos_embeds(
        num_input_tokens: int, vision_hidden_size: int
):
    # pos emb

    pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
    nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

    return pos_emb


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
            self,
            config,
            num_input_tokens: int,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens

        # pos emb
        self.pos_emb = build_pos_embeds(num_input_tokens, config.mm_hidden_size)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        # output = BaseModelOutput(last_hidden_state=x)
        return x

    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #     # # update old ckpt compatible with current code
    #     # pos_emb = state_dict["abstractor.pos_emb"]
    #     # if pos_emb.size(1) == self.pos_emb.size(1) + 1:
    #     #     # remove obsolete first pos emb (for cls token originally)
    #     #     state_dict["abstractor.pos_emb"] = pos_emb[:, 1:]
    #
    #     super()._load_from_state_dict(state_dict, *args, **kwargs)


class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x


class CAbstractor(ConvProjector):
    """C-Abstractor based on RegBlock"""

    def build_net(self):
        encoder_hidden_size = self.config.mm_hidden_size
        hidden_size = 1024
        output_hidden_size = self.config.hidden_size
        depth = 3
        mlp_depth = 2

        n_queries = 25
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.Sequential(s1, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    elif projector_type.startswith('spp'):
        return SPP(config, projector_type)

    elif projector_type == 'ldp':
        return LDPNetProjector(config)

    elif projector_type == 'vanilla':
        return Vanilla(config)

    elif projector_type == 'minigpt':
        return Minigpt(config)

    elif projector_type == 'identity':
        return IdentityMap()

    elif projector_type == 'c-abs':
        return CAbstractor(config, num_input_tokens=729)

    raise ValueError(f'Unknown projector type: {projector_type}')
