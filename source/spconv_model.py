from matplotlib.style import context
import torch
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging
from torch_scatter import scatter_add, scatter_max
from source.model import build_flow, generate_masks

def build_sparse_projections(
    data,
    grid_size=(128, 128),
    spatial_range=None,
):
    coords = data.x
    device = coords.device

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    def to_grid(val, vmin, vmax, size):
        val = (val - vmin) / (vmax - vmin + 1e-6)
        val = (val * (size - 1)).long()
        return torch.clamp(val, 0, size - 1)

    if spatial_range is None:
        spatial_range = {
            "x": (x.min(), x.max()),
            "y": (y.min(), y.max()),
            "z": (z.min(), z.max()),
        }

    H, W = grid_size

    gx = to_grid(x, *spatial_range["x"], W)
    gy = to_grid(y, *spatial_range["y"], H)
    gz = to_grid(z, *spatial_range["z"], H)

    # Features
    feats = torch.ones((coords.shape[0], 1), device=device)

    # ZX
    zx_indices = torch.stack([
        torch.zeros_like(gx),
        gz,
        gx
    ], dim=1).int()

    zx_sparse = spconv.SparseConvTensor(
        features=feats,
        indices=zx_indices,
        spatial_shape=[H, W],
        batch_size=1
    )

    # ZY
    zy_indices = torch.stack([
        torch.zeros_like(gy),
        gz,
        gy
    ], dim=1).int()

    zy_sparse = spconv.SparseConvTensor(
        features=feats,
        indices=zy_indices,
        spatial_shape=[H, W],
        batch_size=1
    )

    return zx_sparse, zy_sparse


class SparseCNNProjectionNetwork(nn.Module):
    def __init__(
        self,
        in_channels=1,
        conv_dims=(8, 16, 32),
        feature_dim=128,
        kernel_size=3,
        padding=1,
        dropout=0.5,
    ):
        super().__init__()

        layers = []
        prev_c = in_channels

        for i, c in enumerate(conv_dims):
            layers.append(
                spconv.SparseSequential(
                    spconv.SubMConv2d(
                        prev_c, c, kernel_size,
                        padding=padding,
                        bias=False,
                        indice_key=f"subm{i}"  
                    ),
                    nn.BatchNorm1d(c),
                    nn.ReLU(),
                )
            )
            prev_c = c

        self.conv = spconv.SparseSequential(*layers)

        self.pool = spconv.SparseGlobalMaxPool()

        self.fc = nn.Sequential(
            nn.Linear(prev_c, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)    

class RegressionCNN(nn.Module):
    def __init__(self, 
        feature_dim: int = 128,
        conv_dims: Tuple=(8, 16, 32),
        fc_dims: Tuple=(128,),
        kernel_size: int=3, 
        padding=1, 
        dropout: float=0.5,
        num_targets: int = 8,
    ):
        super().__init__()

        self.zx_cnn = SparseCNNProjectionNetwork(
            conv_dims=conv_dims,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout
        )

        self.zy_cnn = SparseCNNProjectionNetwork(
            conv_dims=conv_dims,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout
        )

        reg_in_dim = 2 * feature_dim

        layers = []
        for i, dim in enumerate(fc_dims):
            in_dim = reg_in_dim if i == 0 else fc_dims[i - 1]
            layers += [
                nn.Linear(in_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        layers.append(nn.Linear(fc_dims[-1], num_targets))
        self.regressor = nn.Sequential(*layers)

    def forward(self, data):

        zx_sparse, zy_sparse = build_sparse_projections(data)

        zx_features = self.zx_cnn(zx_sparse)
        zy_features = self.zy_cnn(zy_sparse)

        combined = torch.cat([zx_features, zy_features], dim=1)

        return self.regressor(combined)


import torch
import spconv.pytorch as spconv

def build_sparse_3d(
    data,
    grid_size=(64, 64, 64),
    spatial_range=None,
):
    """
    Build a batched 3D SparseConvTensor from PyG Data.

    Args:
        data.x: [N, 3]
        data.batch: [N]
    """

    coords = data.x
    batch = data.batch
    device = coords.device

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # -----------------------------
    # Normalisation → voxel grid
    # -----------------------------
    def to_grid(val, vmin, vmax, size):
        val = (val - vmin) / (vmax - vmin + 1e-6)
        val = (val * (size - 1)).long()
        return torch.clamp(val, 0, size - 1)

    if spatial_range is None:
        spatial_range = {
            "x": (x.min(), x.max()),
            "y": (y.min(), y.max()),
            "z": (z.min(), z.max()),
        }

    Z, Y, X = grid_size

    gx = to_grid(x, *spatial_range["x"], X)
    gy = to_grid(y, *spatial_range["y"], Y)
    gz = to_grid(z, *spatial_range["z"], Z)

    # -----------------------------
    # Indices: [batch, z, y, x]
    # -----------------------------
    indices = torch.stack([batch, gz, gy, gx], dim=1)

    unique_indices, inv = torch.unique(indices, dim=0, return_inverse=True)

    voxel_counts = scatter_add(
        torch.ones_like(inv, dtype=torch.float),
        inv,
    )

    feats = voxel_counts.unsqueeze(1)

    sparse_tensor = spconv.SparseConvTensor(
        features=feats,
        indices=unique_indices.int(),
        spatial_shape=[Z, Y, X],
        batch_size=int(batch.max().item()) + 1
    )

    return sparse_tensor

import torch.nn as nn


# class AdaLN(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.linear = nn.Linear(num_features, num_features * 2)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.linear.reset_parameters()

#     def forward(self, x, c):
#         shift, scale = self.linear(c).chunk(2, dim=-1)
#         return x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)

import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor


class AdaNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], k: float = 0.1, eps: float = 1e-5, bias: bool = False) -> None:
        super(AdaNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.k = k
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = (input - mean).pow(2).mean(dim=-1, keepdim=True) + self.eps
    
        input_norm = (input - mean) * torch.rsqrt(var)
        
        adanorm = self.weight * (1 - self.k * input_norm) * input_norm

        if self.bias is not None:
            adanorm = adanorm + self.bias
    
        return adanorm

class Sparse3DCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        feature_dim=64,
        dropout=0.3,
    ):
        super().__init__()

        self.net = spconv.SparseSequential(

            # Block 1
            spconv.SubMConv3d(in_channels, base_channels, 3, padding=1, indice_key="subm1"),
            # nn.BatchNorm1d(base_channels),
            # nn.LayerNorm(base_channels),
            AdaNorm(base_channels),
            nn.ReLU(),

            # Downsample
            spconv.SparseConv3d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            AdaNorm(base_channels * 2),
            # nn.LayerNorm(base_channels * 2),
            nn.ReLU(),

            # Block 2
            spconv.SubMConv3d(base_channels * 2, base_channels * 2, 3, padding=1, indice_key="subm2"),
            AdaNorm(base_channels * 2),
            # nn.BatchNorm1d(base_channels * 2),
            # nn.LayerNorm(base_channels * 2),  
              
            nn.ReLU(),

            # # Downsample
            spconv.SparseConv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            AdaNorm(base_channels * 4),
            # nn.BatchNorm1d(base_channels * 4),
            # nn.LayerNorm(base_channels * 4),
            nn.ReLU(),

            # Block 3
            spconv.SubMConv3d(base_channels * 4, base_channels * 4, 3, padding=1, indice_key="subm3"),
            # nn.BatchNorm1d(base_channels * 4),
            AdaNorm(base_channels * 4),
            # nn.LayerNorm(base_channels * 4),
            nn.ReLU(),
        )

        self.pool = spconv.SparseGlobalMaxPool()

        self.head = nn.Sequential(
            nn.Linear(base_channels * 4, feature_dim),
            AdaNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return self.head(x)


class Sparse3DRegression(nn.Module):
    def __init__(
        self,
        cfg,
        feature_dim=128,
        fc_dims=(128,),
        num_targets=8,
        dropout=0.3,
    ):
        super().__init__()

        self.backbone = Sparse3DCNN(feature_dim=feature_dim, dropout=dropout)

        layers = []
        for i, dim in enumerate(fc_dims):
            in_dim = feature_dim if i == 0 else fc_dims[i - 1]
            layers += [
                nn.Linear(in_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        layers.append(nn.Linear(fc_dims[-1], num_targets))
        self.regressor = nn.Sequential(*layers)

    def forward(self, data):
        x = build_sparse_3d(data, grid_size=(128*2, 128*2, 128*2))
        feats = self.backbone(x)
        return self.regressor(feats)


class Sparse3DFlowRegression(nn.Module):
    def __init__(
        self,
        cfg,
        feature_dim=64,
        dropout=0.3,
        batch_size=4,
    ):
        super().__init__()

        self.backbone = Sparse3DCNN(feature_dim=feature_dim, dropout=dropout)

        self.context_dim = 64
        self.context_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.context_dim),
        )
        nlayers = 10
        masks = generate_masks(cfg.n_targets, nlayers)
        self.flow = build_flow(y_dim=cfg.n_targets, context_dim=self.context_dim, nlayers=nlayers, masks=masks)

        self.adaln = AdaNorm(self.context_dim)
        self.batch_size = batch_size

    def forward(self, data, y=None):
        
        x = build_sparse_3d(data, grid_size=(128*2, 128*2, 128*2))
        feats = self.backbone(x)
        context = self.context_net(feats)
        context = self.adaln(context)
        
        if y is not None:
            # TRAINING: compute log prob
            log_prob = self.flow.log_prob(inputs=y, context=context)
            return log_prob
        else:
            # INFERENCE: sample predictions
            samples = self.flow.sample(context=context, num_samples=32, batch_size=self.batch_size)
            y_pred = samples.mean(dim=0)
            # return y_pred
            S, B, D = samples.shape
            samples_flat = samples.reshape(S * B, D)

            context_expanded = context.unsqueeze(0).expand(S, B, -1)
            context_flat = context_expanded.reshape(S * B, -1)

            log_probs = self.flow.log_prob(samples_flat, context_flat)
            log_probs = log_probs.view(S, B)

            best_idx = torch.argmax(log_probs, dim=0)
            y_pred_map = samples[best_idx, torch.arange(B)]
            return y_pred_map


    def loss(self, data, y):
        log_prob = self.forward(data, y)
        return -log_prob.mean()


import torch

class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self
    
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDenoiser(nn.Module):
    def __init__(self, y_dim, context_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(y_dim + context_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim)
        )

    def forward(self, y_t, t, context):
        # normalize timestep
        t = t.float().unsqueeze(-1) / 1000.0

        x = torch.cat([y_t, context, t], dim=-1)
        return self.net(x)


def q_sample(y0, t, schedule, noise=None):
    if noise is None:
        noise = torch.randn_like(y0)

    alpha_bar_t = schedule.alpha_bar[t].unsqueeze(-1)

    return (
        torch.sqrt(alpha_bar_t) * y0 +
        torch.sqrt(1 - alpha_bar_t) * noise
    ), noise


class Sparse3DDiffusionRegression(nn.Module):
    def __init__(
        self,
        cfg,
        feature_dim=64,
        dropout=0.3,
        batch_size=4,
    ):
        super().__init__()

        self.backbone = Sparse3DCNN(feature_dim=feature_dim, dropout=dropout)

        self.context_dim = 64
        self.context_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.context_dim),
        )
    
        self.diffusion = ConditionalDenoiser(y_dim=cfg.n_targets, context_dim=self.context_dim, hidden_dim=128)
        self.schedule = DiffusionSchedule(T=1000, beta_start=1e-4, beta_end=0.02)

        self.adaln = AdaNorm(self.context_dim)
        self.batch_size = batch_size

    def forward(self, data, y0=None):
        
        x = build_sparse_3d(data, grid_size=(128*2, 128*2, 128*2))
        feats = self.backbone(x)
        context = self.context_net(feats)
        context = self.adaln(context)

        if y0 is not None:
            B = y0.size(0)
            device = y0.device

            t = torch.randint(0, self.schedule.T, (B,), device=device)

            y_t, noise = q_sample(y0, t, self.schedule)

            noise_pred = self.diffusion(y_t, t, context)

            return F.mse_loss(noise_pred, noise)

        else:
            return self.sample(context)

        

    @torch.no_grad()
    def sample(self, context):
        B, y_dim = context.size(0), self.diffusion.net[-1].out_features
        device = self.device

        y_t = torch.randn(B, y_dim, device=device)

        for t in reversed(range(self.schedule.T)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            noise_pred = self.diffusion(y_t, t_tensor, context)

            alpha = self.schedule.alpha[t]
            alpha_bar = self.schedule.alpha_bar[t]
            beta = self.schedule.beta[t]

            if t > 0:
                noise = torch.randn_like(y_t)
            else:
                noise = 0

            y_t = (
                (1 / torch.sqrt(alpha)) *
                (y_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred)
                + torch.sqrt(beta) * noise
            )

        return y_t


import torch
import torch.nn as nn
import spconv.pytorch as spconv


def conv_block(in_c, out_c, kernel=3, stride=1, padding=1):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_c, out_c, kernel, padding=padding, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )


def down_block(in_c, out_c, key):
    return spconv.SparseSequential(
        spconv.SparseConv3d(
            in_c, out_c, 3,
            stride=2,
            padding=1,
            bias=False,
            indice_key=key
        ),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
        conv_block(out_c, out_c),
    )


def up_block(in_c, out_c, key):
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(
            in_c, out_c, 3,
            bias=False,
            indice_key=key
        ),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )


class SparseUNet4(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        # -------- Encoder --------
        self.enc1 = conv_block(in_channels, c1)     # no downsample
        self.enc2 = down_block(c1, c2, key="down2")
        self.enc3 = down_block(c2, c3, key="down3")
        self.enc4 = down_block(c3, c4, key="down4")

        # Bottleneck
        self.bottleneck = down_block(c4, c5, key="down5")

        # -------- Decoder --------
        self.up4 = up_block(c5, c4, key="down5")
        self.dec4 = conv_block(c4 + c4, c4)

        self.up3 = up_block(c4, c3, key="down4")
        self.dec3 = conv_block(c3 + c3, c3)

        self.up2 = up_block(c3, c2, key="down3")
        self.dec2 = conv_block(c2 + c2, c2)

        self.up1 = up_block(c2, c1, key="down2")
        self.dec1 = conv_block(c1 + c1, c1)

        # Output head
        self.head = spconv.SubMConv3d(c1, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(x)   # highest resolution
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        # -------- Decoder --------
        d4 = self.up4(b)
        d4 = self._concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._concat(d1, e1)
        d1 = self.dec1(d1)

        return d1
       
    def _concat(self, a, b):
        """
        Concatenate sparse tensors (features only).
        Assumes same coordinates (true after inverse conv).
        """
        assert (a.indices == b.indices).all(), "Sparse coords mismatch!"

        out = a.replace_feature(
            torch.cat([a.features, b.features], dim=1)
        )
        return out


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.pt_nu_head  = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.eta_nu_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.phi_nu_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.E_nu_head   = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.pt_lep_head  = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 

        self.eta_lep_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.phi_lep_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # sin(phi), cos(phi)
        )

        self.E_lep_head   = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


        self.E_jet_head   = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pt_jet_head   = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.eta_jet_head   = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.phi_jet_head   = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # sin(phi), cos(phi)
        )

        self.pt_miss_head   = nn.Sequential(
            nn.Linear(in_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.phi_miss_head   = nn.Sequential(
            nn.Linear(in_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # sin(phi), cos(phi)
        )
        self.E_miss_head   = nn.Sequential(
            nn.Linear(in_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.eta_miss_head   = nn.Sequential(
            nn.Linear(in_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        
        pT_lep = self.pt_lep_head(x).squeeze(-1)
        eta_lep = self.eta_lep_head(x).squeeze(-1)
        sincos_phi_lep = self.phi_lep_head(x).squeeze(-1)
        E_lep = self.E_lep_head(x).squeeze(-1)

        lep_pred = torch.stack([pT_lep, eta_lep, E_lep], dim=-1)
        miss_input = torch.cat([x, lep_pred.detach()], dim=-1)

        pT_miss = self.pt_miss_head(miss_input).squeeze(-1)
        eta_miss = self.eta_miss_head(miss_input).squeeze(-1)
        sincos_phi_miss = self.phi_miss_head(miss_input)
        E_miss = self.E_miss_head(miss_input).squeeze(-1)


        result = {
            "pT_nu":  self.pt_nu_head(x).squeeze(-1),
            "eta_nu": self.eta_nu_head(x).squeeze(-1),
            "sincos_phi_nu": self.phi_nu_head(x),
            "E_nu":   self.E_nu_head(x).squeeze(-1),
            
            "pT_lep":  pT_lep,
            "eta_lep": eta_lep,
            "sincos_phi_lep": sincos_phi_lep,
            "E_lep":   E_lep,

            "pT_jet":  self.pt_jet_head(x).squeeze(-1),
            "eta_jet": self.eta_jet_head(x).squeeze(-1),
            "sincos_phi_jet": self.phi_jet_head(x),
            "E_jet":   self.E_jet_head(x).squeeze(-1),

            "pT_miss":  pT_miss,
            "eta_miss": eta_miss,
            "sincos_phi_miss": sincos_phi_miss,
            "E_miss":   E_miss,
        }

        result["phi_nu"] = torch.atan2(result["sincos_phi_nu"][:, 0], result["sincos_phi_nu"][:, 1])
        result["phi_lep"] = torch.atan2(result["sincos_phi_lep"][:, 0], result["sincos_phi_lep"][:, 1])
        result["phi_jet"] = torch.atan2(result["sincos_phi_jet"][:, 0], result["sincos_phi_jet"][:, 1])
        result["phi_miss"] = torch.atan2(result["sincos_phi_miss"][:, 0], result["sincos_phi_miss"][:, 1])

        return result

from torch_scatter import scatter_mean

# class FullModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.unet = SparseUNet4(in_channels=1, base_channels=16)
#         self.head = MultiTaskHead(in_dim=16)  # final UNet channels

#     def forward(self, data):
#         x = build_sparse_3d(data, grid_size=(128*2, 128*2, 128*2))
#         x = self.unet(x)

#         feats = x.features
#         batch = x.indices[:, 0].long()

#         global_feat = scatter_mean(feats, batch, dim=0)

#         return self.head(global_feat)
    
def loss_fn(pred, target):
    loss_pt  = F.mse_loss(pred["pT_nu"],  target["pT_nu"])
    loss_eta = F.mse_loss(pred["eta_nu"], target["eta_nu"])
    loss_E   = F.mse_loss(pred["E_nu"],   target["E_nu"])
    loss_phi_nu = F.mse_loss(pred["sincos_phi_nu"], torch.stack([target["sin_phi_nu"], target["cos_phi_nu"]], dim=-1))

    loss_pt_lep  = F.mse_loss(pred["pT_lep"],  target["pT_lep"])
    loss_eta_lep = F.mse_loss(pred["eta_lep"], target["eta_lep"])
    loss_E_lep   = F.mse_loss(pred["E_lep"],   target["E_lep"])
    loss_phi_lep = F.mse_loss(pred["sincos_phi_lep"], torch.stack([target["sin_phi_lep"], target["cos_phi_lep"]], dim=-1))

    loss_pt_jet  = F.mse_loss(pred["pT_jet"],  target["pT_jet"])
    loss_eta_jet = F.mse_loss(pred["eta_jet"], target["eta_jet"])
    loss_E_jet   = F.mse_loss(pred["E_jet"],   target["E_jet"])
    loss_phi_jet = F.mse_loss(pred["sincos_phi_jet"], torch.stack([target["sin_phi_jet"], target["cos_phi_jet"]], dim=-1))

    loss_miss_pt  = F.mse_loss(pred["pT_miss"],  target["pT_miss"])
    loss_miss_eta = F.mse_loss(pred["eta_miss"], target["eta_miss"])
    loss_miss_E   = F.mse_loss(pred["E_miss"],   target["E_miss"])
    loss_miss_phi = F.mse_loss(pred["sincos_phi_miss"], torch.stack([target["sin_phi_miss"], target["cos_phi_miss"]], dim=-1))

    return (
        1.0 * loss_pt +
        1.0 * loss_eta +
        1.0 * loss_phi_nu +
        1.0 * loss_E +
        1.0 * loss_pt_lep +
        1.0 * loss_eta_lep +
        1.0 * loss_phi_lep +
        1.0 * loss_E_lep + 
        1.0 * loss_pt_jet +
        1.0 * loss_eta_jet +
        1.0 * loss_phi_jet +
        1.0 * loss_E_jet +
        1.0 * loss_miss_pt +
        1.0 * loss_miss_eta +
        1.0 * loss_miss_phi +
        1.0 * loss_miss_E
    )

class SpatialPyramidAggregator(nn.Module):
    """
    Multi-scale spatial pooling for sparse 3D tensors.
    
    Pools features at multiple spatial scales by dividing the detector
    volume into progressively coarser grids, then concatenates all scales.
    This preserves directional/spatial structure that scatter_mean destroys.
    
    For a neutrino detector, coarse scales capture global event topology
    (beam direction, overall energy flow) while fine scales capture local
    shower structure.
    """
    def __init__(self, in_dim, grid_size=(256, 256, 256)):
        super().__init__()
        self.grid_size = grid_size
        
        # Subdivisions along each axis at each pyramid level
        # Level 0: 1x1x1   — pure global mean (baseline)
        # Level 1: 2x2x2   — 8 coarse spatial bins
        # Level 2: 4x4x4   — 64 medium spatial bins  
        # Level 3: 4x4x8   — asymmetric: finer along z (beam axis)
        self.levels = [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (4, 4, 8),   # z is beam axis — resolve it more finely
        ]
        
        # Each level produces in_dim features per bin, all concatenated
        # then projected down to a fixed output dim
        n_bins_total = sum(nx*ny*nz for nx,ny,nz in self.levels)
        self.proj = nn.Sequential(
            nn.Linear(in_dim * n_bins_total, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # Attention network — scores each voxel's contribution
        # before pooling, so the network can focus on the most
        # informative hits (e.g. track endpoints, shower cores)
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # scalar weight per voxel
        )

    def forward(self, feats, batch_indices, voxel_coords, batch_size):
        """
        Args:
            feats:        [N, C]   sparse voxel features
            batch_indices:[N]      which event each voxel belongs to
            voxel_coords: [N, 3]   integer voxel coordinates (x, y, z)
            batch_size:   int      number of events in batch
        """
        device = feats.device
        C = feats.shape[1]
        
        # Compute per-voxel attention weights within each event
        raw_attn = self.attention(feats)  # [N, 1]
        
        # Softmax within each event separately
        attn_weights = self._event_softmax(raw_attn, batch_indices, batch_size)  # [N, 1]
        weighted_feats = feats * attn_weights  # [N, C]
        
        level_pools = []
        
        for (nx, ny, nz) in self.levels:
            # Map voxel coords into bin indices for this level
            # voxel_coords are in [0, grid_size), bin them into [0, n_divisions)
            bx = (voxel_coords[:, 0].float() / self.grid_size[0] * nx).long().clamp(0, nx-1)
            by = (voxel_coords[:, 1].float() / self.grid_size[1] * ny).long().clamp(0, ny-1)
            bz = (voxel_coords[:, 2].float() / self.grid_size[2] * nz).long().clamp(0, nz-1)
            
            # Flatten spatial bin + batch into a single index for scatter
            # bin_id is unique per (event, spatial_bin)
            bin_id = (batch_indices * nx * ny * nz +
                      bx * ny * nz +
                      by * nz +
                      bz)  # [N]
            
            n_bins = batch_size * nx * ny * nz
            
            # Attention-weighted pool into spatial bins
            pooled = scatter_mean(weighted_feats, bin_id, dim=0, out=torch.zeros(n_bins, C, device=device))
            # pooled: [batch_size * nx*ny*nz, C]
            
            # Reshape to [batch_size, nx*ny*nz*C] and append
            pooled = pooled.view(batch_size, nx * ny * nz * C)
            level_pools.append(pooled)
        
        # Concatenate all pyramid levels: [batch_size, n_bins_total * C]
        pyramid = torch.cat(level_pools, dim=-1)
        
        # Project down to fixed output dim
        return self.proj(pyramid)  # [batch_size, 256]

    def _event_softmax(self, scores, batch_indices, batch_size):
        """Softmax over voxels within each event independently."""
        s = scores.squeeze(-1)  # [N]
        
        # scatter_max returns (values, argmax) — need [0]
        scores_max = scatter_max(s, batch_indices, dim=0, dim_size=batch_size)[0]  # [B]
        scores_shifted = s - scores_max[batch_indices]
        
        scores_exp = torch.exp(scores_shifted)
        
        # scatter_add returns a tensor directly — no [0]
        scores_sum = scatter_add(scores_exp, batch_indices, dim=0, dim_size=batch_size)  # [B]
        
        weights = scores_exp / (scores_sum[batch_indices] + 1e-8)
        
        return weights.unsqueeze(-1)  # [N, 1]
        

from torch_scatter import scatter_mean, scatter_max, scatter_add

class FullModel(nn.Module):
    def __init__(self, grid_size=(256, 256, 256)):
        super().__init__()
        self.grid_size = grid_size
        self.unet = SparseUNet4(in_channels=1, base_channels=16)
        self.aggregator = SpatialPyramidAggregator(in_dim=16, grid_size=grid_size)
        
        miss_in_dim = 256 + 3   # aggregator output + tau conditioning
        self.head = MultiTaskHead(in_dim=256)

    def forward(self, data):
        x = build_sparse_3d(data, grid_size=self.grid_size)
        x = self.unet(x)

        feats = x.features          # [N, 16]
        batch = x.indices[:, 0].long()
        coords = x.indices[:, 1:]   # [N, 3]  — drop batch dim
        batch_size = batch.max().item() + 1

        global_feat = self.aggregator(feats, batch, coords, batch_size)
        # global_feat: [batch_size, 256]

        return self.head(global_feat)
    
class FullFlowModel(nn.Module):
    def __init__(self, cfg, batch_size=4):
        super().__init__()
        self.unet = SparseUNet4(in_channels=1, base_channels=16)
        
        self.context_dim = 16

        ntargets = 16
        nlayers = 5
        masks = generate_masks(ntargets, nlayers)
        self.flow = build_flow(y_dim=ntargets, context_dim=self.context_dim, nlayers=nlayers, masks=masks)
        self.batch_size = batch_size

    def forward(self, data, y=None):
        x = build_sparse_3d(data, grid_size=(128*2, 128*2, 128*2))
        x = self.unet(x)

        feats = x.features
        batch = x.indices[:, 0].long()

        global_feat = scatter_mean(feats, batch, dim=0)

        if y is not None:
            # TRAINING: compute log prob
            log_prob = self.flow.log_prob(inputs=y, context=global_feat)
            return log_prob

        else:
            # INFERENCE: sample predictions
            samples = self.flow.sample(context=global_feat, num_samples=32, batch_size=self.batch_size)
            y_pred = samples.mean(dim=0)
            # return y_pred
            S, B, D = samples.shape
            samples_flat = samples.reshape(S * B, D)

            context_expanded = global_feat.unsqueeze(0).expand(S, B, -1)
            context_flat = context_expanded.reshape(S * B, -1)

            log_probs = self.flow.log_prob(samples_flat, context_flat)
            log_probs = log_probs.view(S, B)

            best_idx = torch.argmax(log_probs, dim=0)
            y_pred_map = samples[best_idx, torch.arange(B)]
            return y_pred_map


    def loss(self, data, y):
        log_prob = self.forward(data, y)
        return -log_prob.mean()
