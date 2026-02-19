import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GravNetConv, global_mean_pool, global_max_pool


# class GravNetModel(nn.Module):
#     def __init__(
#         self,
#         input_dim=3,
#         hidden_dim=64,
#         grav_dim=4,
#         k=16,
#         num_classes=4,
#     ):
#         super().__init__()

#         self.input_mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#         )

#         self.grav1 = GravNetConv(
#             hidden_dim,
#             hidden_dim,
#             space_dimensions=grav_dim,
#             propagate_dimensions=hidden_dim,
#             k=k,
#         )

#         self.grav2 = GravNetConv(
#             hidden_dim,
#             hidden_dim,
#             space_dimensions=grav_dim,
#             propagate_dimensions=hidden_dim,
#             k=k,
#         )

#         self.grav3 = GravNetConv(
#             hidden_dim,
#             hidden_dim,
#             space_dimensions=grav_dim,
#             propagate_dimensions=hidden_dim,
#             k=k,
#         )

#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)

#         self.event_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#         )

#         self.class_head = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes),
#         )

#         self.energy_head = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )

#     def forward(self, data):
#         x, batch = data.x, data.batch

#         x = self.input_mlp(x)

#         x = F.relu(self.bn1(self.grav1(x, batch)))
#         x = F.relu(self.bn2(self.grav2(x, batch)))
#         x = F.relu(self.bn3(self.grav3(x, batch)))

#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         event_repr = torch.cat([x_mean, x_max], dim=1)

#         event_repr = self.event_mlp(event_repr)

#         class_logits = self.class_head(event_repr)
#         energy = self.energy_head(event_repr).squeeze(-1)

#         return class_logits, energy



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GravNetConv, global_mean_pool
import pytorch_lightning as pl

class GravNetBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, space_dimensions=4, propagate_dimensions=32, k=16):
        super().__init__()
        self.gravnet = GravNetConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x_in = x
        x = self.gravnet(x, edge_index)
        x = self.lin(x)
        x = self.norm(x)
        x = self.act(x)
        return x + x_in  # residual

class GravNetModel(nn.Module):
    def __init__(self, node_features=3, hidden_dim=64, n_classes=4):
        super().__init__()
        self.input_lin = nn.Linear(node_features, hidden_dim)

        # Stack of GravNet blocks
        self.blocks = nn.ModuleList([
            GravNetBlock(hidden_dim, hidden_dim, k=32),
            GravNetBlock(hidden_dim, hidden_dim, k=16),
            GravNetBlock(hidden_dim, hidden_dim, k=16)
        ])

        # Event-level MLP
        self.event_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Classification head
        self.cls_head = nn.Linear(128, n_classes)
        # Energy regression head
        self.energy_head = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_lin(x)
        for block in self.blocks:
            x = block(x, edge_index, batch)

        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.event_mlp(x)

        class_logits = self.cls_head(x)
        energy_pred = self.energy_head(x).squeeze(-1)
        return class_logits, energy_pred

# class GravNetLightning(pl.LightningModule):
#     def __init__(self, lr=1e-3, n_classes=4):
#         super().__init__()
#         self.model = GravNetEventModel(n_classes=n_classes)
#         self.lr = lr

#         self.cls_loss = nn.CrossEntropyLoss()
#         self.reg_loss = nn.MSELoss()

#     def forward(self, data):
#         return self.model(data)

#     def training_step(self, batch, batch_idx):
#         class_logits, energy_pred = self(batch)
#         cls_loss = self.cls_loss(class_logits, batch.y_class)
#         reg_loss = self.reg_loss(energy_pred, batch.y_energy)
#         loss = cls_loss + reg_loss
#         self.log('train_loss', loss)
#         self.log('train_cls_loss', cls_loss)
#         self.log('train_reg_loss', reg_loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         class_logits, energy_pred = self(batch)
#         cls_loss = self.cls_loss(class_logits, batch.y_class)
#         reg_loss = self.reg_loss(energy_pred, batch.y_energy)
#         loss = cls_loss + reg_loss
#         self.log('val_loss', loss)
#         self.log('val_cls_loss', cls_loss)
#         self.log('val_reg_loss', reg_loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer
