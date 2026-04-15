import torch
import torch.nn as nn
import spconv.pytorch as spconv
from source.spconv_model import AdaNorm

def subsample_tokens(features, batch_idx, max_tokens=1024):
    keep = []

    B = int(batch_idx.max().item()) + 1

    for b in range(B):
        idx = (batch_idx == b).nonzero(as_tuple=False).squeeze()

        # ✅ FIX: ensure 1D tensor
        if idx.ndim == 0:
            idx = idx.unsqueeze(0)

        if idx.numel() > max_tokens:
            perm = torch.randperm(idx.numel(), device=idx.device)
            idx = idx[perm[:max_tokens]]

        keep.append(idx)

    keep = torch.cat(keep)

    return features[keep], batch_idx[keep]

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

        self.token_pool = nn.Linear(base_channels*4, 128)
        self.global_pool = spconv.SparseGlobalMaxPool()
        self.fc = nn.Linear(base_channels*4, 256)

    # def forward(self, sparse_tensor):
    #     x = self.net(sparse_tensor)              # sparse features
    #     x = self.global_pool(x)                  # [B, C]
    #     x = self.fc(x)                           # context vector
    #     return x
    def forward(self, sparse_tensor):
        x = self.net(sparse_tensor)   # SparseConvTensor

        features = x.features         # [N_active, C]
        features = self.token_pool(features)
        batch_idx = x.indices[:, 0]   # [N_active]

        return features, batch_idx
    

def timestep_embedding(t, dim=64):
    device = t.device
    half = dim // 2

    freqs = torch.exp(
        -torch.arange(half, device=device) * torch.log(torch.tensor(10000.0)) / half
    )

    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class CrossAttentionBlock(nn.Module):
    def __init__(self, y_dim, context_dim, n_heads=4):
        super().__init__()

        self.query = nn.Linear(y_dim, context_dim)
        self.key   = nn.Linear(context_dim, context_dim)
        self.value = nn.Linear(context_dim, context_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=n_heads,
            batch_first=True
        )

        self.out = nn.Linear(context_dim, y_dim)

    def forward(self, y, context, mask=None):
        """
        y: [B, y_dim]
        context: [B, N_tokens, C]
        """

        q = self.query(y).unsqueeze(1)   # [B, 1, C]
        k = self.key(context)            # [B, N, C]
        v = self.value(context)          # [B, N, C]

        attn_out, _ = self.attn(q, k, v, key_padding_mask=mask)

        return self.out(attn_out.squeeze(1))

def to_dense_batch(features, batch_idx):
    B = batch_idx.max().item() + 1

    counts = torch.bincount(batch_idx)
    N_max = counts.max()

    C = features.size(1)

    out = features.new_zeros(B, N_max, C)
    mask = torch.ones(B, N_max, dtype=torch.bool, device=features.device)

    for b in range(B):
        idx = (batch_idx == b)
        n = idx.sum()

        out[b, :n] = features[idx]
        mask[b, :n] = False   # False = valid token

    return out, mask

# class DiffusionHead(nn.Module):
#     def __init__(self, y_dim, context_dim=256, hidden_dim=512):
#         super().__init__()

#         self.time_mlp = nn.Sequential(
#             nn.Linear(64, hidden_dim),
#             nn.ReLU()
#         )

#         self.net = nn.Sequential(
#             nn.Linear(y_dim + context_dim + hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, y_dim)
#         )

#     def forward(self, y_t, t, context):
#         t_emb = timestep_embedding(t, 64)
#         t_emb = self.time_mlp(t_emb)

#         x = torch.cat([y_t, context, t_emb], dim=-1)
#         return self.net(x)

class DiffusionHead(nn.Module):
    def __init__(self, y_dim, context_dim=128, hidden_dim=512):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        self.cross_attn = CrossAttentionBlock(
            y_dim=hidden_dim,
            context_dim=context_dim
        )
        # self.cross_attn2 = CrossAttentionBlock(
        #     y_dim=hidden_dim,
        #     context_dim=context_dim
        # )

        # self.attn = nn.MultiheadAttention(256, 4, batch_first=True)

        self.input_proj = nn.Linear(y_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t, t, context_tokens, mask):
        t_emb = timestep_embedding(t, 64)
        t_emb = self.time_mlp(t_emb)

        y = self.input_proj(y_t) + t_emb

        y = self.cross_attn1(y, context_tokens, mask)
        # y = self.cross_attn2(y, context_tokens, mask)
        
        # y, _ = self.attn(y.unsqueeze(1), context_tokens, context_tokens, key_padding_mask=mask)
        # y = y.squeeze(1)

        return self.output(y)    

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
        features, batch_idx = self.encoder(sparse_tensor)

        context, mask = to_dense_batch(features, batch_idx)

        noise_pred = self.diffusion(y_t, t, context, mask)

        return noise_pred

        