import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv_model import AdaNorm

class SparseEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=32):
        super().__init__()

        self.net = spconv.SparseSequential(
            # Input: SparseConvTensor

            spconv.SubMConv3d(input_channels, base_channels, 3, padding=1, indice_key="subm1"),
            AdaNorm(base_channels),
            nn.ReLU(),

            spconv.SubMConv3d(base_channels, base_channels, 3, padding=1, indice_key="subm1"),
            AdaNorm(base_channels),
            nn.ReLU(),

            # Downsample
            spconv.SparseConv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            AdaNorm(base_channels*2),
            nn.ReLU(),

            spconv.SubMConv3d(base_channels*2, base_channels*2, 3, padding=1, indice_key="subm2"),
            AdaNorm(base_channels*2),
            nn.ReLU(),

            # Another downsample
            spconv.SparseConv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            AdaNorm(base_channels*4),
            nn.ReLU(),
        )

        self.global_pool = spconv.SparseGlobalMaxPool()
        self.fc = nn.Linear(base_channels*4, 256)

    def forward(self, sparse_tensor):
        x = self.net(sparse_tensor)              # sparse features
        x = self.global_pool(x)                  # [B, C]
        x = self.fc(x)                           # context vector
        return x
    

def timestep_embedding(t, dim=64):
    device = t.device
    half = dim // 2

    freqs = torch.exp(
        -torch.arange(half, device=device) * torch.log(torch.tensor(10000.0)) / half
    )

    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionHead(nn.Module):
    def __init__(self, y_dim, context_dim=256, hidden_dim=512):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            nn.Linear(y_dim + context_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim)
        )

    def forward(self, y_t, t, context):
        t_emb = timestep_embedding(t, 64)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([y_t, context, t_emb], dim=-1)
        return self.net(x)
    

class DiffusionSchedule:
    def __init__(self, T=1000):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def q_sample(y0, t, schedule, noise=None):
    if noise is None:
        noise = torch.randn_like(y0)

    alpha_bar_t = schedule.alpha_bar[t].unsqueeze(-1)

    return (
        torch.sqrt(alpha_bar_t) * y0 +
        torch.sqrt(1 - alpha_bar_t) * noise
    ), noise


class SparseDiffusionModel(nn.Module):
    def __init__(self, y_dim, input_channels=1):
        super().__init__()

        self.encoder = SparseEncoder(input_channels)
        self.diffusion = DiffusionHead(y_dim)

    def forward(self, sparse_tensor, y_t, t):
        context = self.encoder(sparse_tensor)
        noise_pred = self.diffusion(y_t, t, context)
        return noise_pred