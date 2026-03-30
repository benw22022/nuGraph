import torch
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging
from torch_scatter import scatter_add
from source.model import build_flow

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
                        indice_key=f"subm{i}"   # ✅ important
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
        feature_dim=128,
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
        x = build_sparse_3d(data, grid_size=(128, 128, 64))
        feats = self.backbone(x)
        return self.regressor(feats)


class Sparse3DFlowRegression(nn.Module):
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

        self.context_dim = 256
        self.context_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.context_dim),
        )
        
        self.flow = build_flow(y_dim=cfg.n_targets, context_dim=self.context_dim)

        self.adaln = AdaNorm(self.context_dim)

    def forward(self, data, y=None):
        
        x = build_sparse_3d(data, grid_size=(128, 128, 64))
        feats = self.backbone(x)
        context = self.context_net(feats)
        context = self.adaln(context)
        
        if y is not None:
            # TRAINING: compute log prob
            log_prob = self.flow.log_prob(inputs=y, context=context)
            return log_prob
        else:
            # INFERENCE: sample predictions
            samples = self.flow.sample(context=context, num_samples=100, batch_size=4)
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
