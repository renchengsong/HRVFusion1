import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from math import sqrt
from argparse import ArgumentParser
from params import params
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self,n_specs):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 3], stride=[1, 1], padding=[1, 1]) #wsst


  def forward(self, x):
      x = torch.unsqueeze(x, 1)
      x = self.conv1(x)
      x = F.leaky_relu(x, 0.4)
      x = torch.squeeze(x, 1)
      return x

class Frequencydomain_FFN(nn.Module):
  def __init__(self, dim, mlp_ratio):
    super().__init__()

    self.scale = 0.02
    self.dim = dim * mlp_ratio

    self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
    self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
    self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
    self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

    self.fc1 = nn.Sequential(
      nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),
      nn.BatchNorm1d(dim * mlp_ratio),
      nn.ReLU(),
    )
    self.fc2 = nn.Sequential(
      nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),
      nn.BatchNorm1d(dim),
    )


  def forward(self, x):
    B, N, C = x.shape

    x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

    x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

    x_real = F.relu(
      torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
      torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
      self.rb
    )
    x_imag = F.relu(
      torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
      torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
      self.ib
    )

    x_fre = torch.stack([x_real, x_imag], dim=-1).float()
    x_fre = torch.view_as_complex(x_fre)
    x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
    x = x.real.float()

    x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
    return x

class MambaLayer(nn.Module):
  def __init__(self, dim, d_state=64, d_conv=4, expand=2):
    super().__init__()
    self.dim = dim
    self.norm = nn.LayerNorm(dim)
    self.mamba = Mamba(
      d_model=dim,
      d_state=d_state,
      d_conv=d_conv,
      expand=expand
    )
  def forward(self, x):
    B, N, C = x.shape
    x_norm = self.norm(x)
    x_mamba = self.mamba(x_norm)
    return x_mamba

class Block_mamba(nn.Module):
  def __init__(self,
               dim,
               mlp_ratio,
               drop_path=0.,
               norm_layer=nn.LayerNorm,
               ):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.norm2 = norm_layer(dim)
    self.attn = MambaLayer(dim)
    self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
      fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      fan_out //= m.groups
      m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
      if m.bias is not None:
        m.bias.data.zero_()

  def forward(self, x):
    B, D, C = x.size()
    x_path = torch.zeros_like(x)

    x_o = self.drop_path(self.attn(x))

    tt = D // 2
    for j in range(D//tt):
      x_div = self.attn(x[:,j * tt : (j + 1)* tt,:])
      x_path[:,j * tt : (j + 1)* tt,:] = x_div
    x_o = x_o + self.drop_path(x_path)

    tt = D // 4
    for j in range(D//tt):
      x_div = self.attn(x[:,j * tt : (j + 1)* tt,:])
      x_path[:,j * tt : (j + 1)* tt,:] = x_div
    x_o = x_o + self.drop_path(x_path)

    tt = D // 8
    for j in range(D//tt):
      x_div = self.attn(x[:,j * tt : (j + 1)* tt,:])
      x_path[:,j * tt : (j + 1)* tt,:] = x_div
    x_o = x_o + self.drop_path(x_path)

    x = x + self.drop_path(self.norm1(x_o))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
  if isinstance(module, nn.Linear):
    if module.bias is not None:
      if not getattr(module.bias, "_no_reinit", False):
        nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    nn.init.normal_(module.weight, std=initializer_range)

  if rescale_prenorm_residual:
    for name, p in module.named_parameters():
      if name in ["out_proj.weight", "fc2.weight"]:
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        with torch.no_grad():
          p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
  if isinstance(m, nn.Linear):
    trunc_normal_(m.weight, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.Conv2d):
    # NOTE conv was left to pytorch default in my original init
    lecun_normal_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
    nn.init.zeros_(m.bias)
    nn.init.ones_(m.weight)

class FeatureFusion(nn.Module):
  def __init__(self, n_specs, residual_channels, dilation):
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.conditioner_projection = Conv1d(n_specs, 2 * residual_channels, 1)

  def forward(self, x, conditioner, diffusion_step):
    #x [32 64 241]
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    conditioner = self.conditioner_projection(conditioner)
    y = x + diffusion_step
    y = self.dilated_conv(y) + conditioner
    return y


class TemporalCNNBlock(nn.Module):
  def __init__(self, dim, kernel_size=5, dilation=1, dropout=0.1):
    super().__init__()
    padding = (kernel_size - 1) // 2 * dilation
    self.conv = nn.Sequential(
      nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
      nn.BatchNorm1d(dim),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    # x: [B, T, C] -> [B, C, T] -> conv -> [B, C, T] -> [B, T, C]
    x = x.transpose(1, 2)
    x = self.conv(x)
    x = x.transpose(1, 2)
    return x

class DiffuSE(nn.Module):
  def __init__(self, args, params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    self.spectrogram_upsampler = SpectrogramUpsampler(params.n_specs)
    self.feature_fusion = FeatureFusion(
      params.n_specs,
      params.residual_channels,
      dilation=1
    )
    dpr = [x.item() for x in torch.linspace(0, params.drop_path_rate, params.depth)]  # stochastic depth decay rule
    inter_dpr = [0.0] + dpr
    self.blocks = nn.ModuleList([Block_mamba(
      dim = params.embed_dim,
      mlp_ratio = params.mlp_ratio,
      drop_path=inter_dpr[i],
      norm_layer=nn.LayerNorm,)
      for i in range(params.depth)])

    self.apply(segm_init_weights)
    self.apply(
      partial(
        _init_weights,
        n_layer=params.depth,
        **(params.initializer_cfg if params.initializer_cfg is not None else {}),
      )
    )
    self.skip_projection = Conv1d(2*params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, gt, spectrogram, diffusion_step):
    x = gt.unsqueeze(1)
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    spectrogram = self.spectrogram_upsampler(spectrogram)
    x = self.feature_fusion(x, spectrogram, diffusion_step)
    x = x.transpose(1, 2)

    for blk in self.blocks:
      x = blk(x)

    x=  x.transpose(1, 2)
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x

if __name__ == '__main__':
  parser = ArgumentParser(description='TEST MODEL')
  parser.add_argument('--chrom_path', default='data/test/chrom/sample.npy',
                      help='input noisy wav directory')
  parser.add_argument('--spectrogram', nargs='+', default=['data/test/spectrogram/sample.spec.npy'],
                      help='space separated list of directories from spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--max_steps', default=50, type=int,
                      help='maximum number of training steps')
  args = parser.parse_args()
  diffuse = DiffuSE(args, params=params)
  print(diffuse)

