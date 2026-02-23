import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from torch_geometric.nn import GravNetConv, global_mean_pool, global_max_pool
from torch_geometric.nn import global_mean_pool, global_max_pool


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GravNetConv, global_mean_pool
import pytorch_lightning as pl
from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, knn_graph, radius_graph
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=True,
    dropout=0.0,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers with dropout
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


class GravConv(nn.Module):
    def __init__(self, hparams, input_size=None, output_size=None):
        super().__init__()
        self.hparams = hparams
        self.feature_dropout = hparams["feature_dropout"] if "feature_dropout" in hparams else 0.0
        self.spatial_dropout = hparams["spatial_dropout"] if "spatial_dropout" in hparams else 0.0
        self.input_size = hparams["hidden"] if input_size is None else input_size
        self.output_size = hparams["hidden"] if output_size is None else output_size
        

        self.feature_network = make_mlp(
                2*(self.input_size + 1),
                [self.output_size] * hparams["nb_node_layer"],
                output_activation=hparams["hidden_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.feature_dropout
        )

        self.spatial_network = make_mlp(
                self.input_size + 1,
                [self.input_size] * hparams["nb_node_layer"] + [hparams["emb_dims"]],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.spatial_dropout
        )

        # This handles the various r, k, and random edge options
        self.setup_neighborhood_configuration()

    def get_neighbors(self, spatial_features):
        
        edge_index = torch.empty([2, 0], dtype=torch.int64, device=spatial_features.device)
 
        if self.use_radius:
            radius_edges = radius_graph(spatial_features, r=self.r, max_num_neighbors=self.hparams["max_knn"], batch=self.batch, loop=self.hparams["self_loop"])
            edge_index = torch.cat([edge_index, radius_edges], dim=1)
        
        if self.use_knn and self.knn > 0:
            k_edges = knn_graph(spatial_features, k=self.knn, batch=self.batch, loop=True)
            edge_index = torch.cat([edge_index, k_edges], dim=1)

        if self.use_rand_k and self.rand_k > 0:
            random_edges = knn_graph(torch.rand(spatial_features.shape[0], 2, device=spatial_features.device), k=self.rand_k, batch=self.batch, loop=True) 
            edge_index = torch.cat([edge_index, random_edges], dim=1)
        
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def get_grav_function(self, d):
        grav_weight = self.grav_weight
        grav_function = - grav_weight * d / self.r**2
        
        return grav_function, grav_weight

    def get_attention_weight(self, spatial_features, hidden_features, edge_index):
        start, end = edge_index
        d = torch.sum((spatial_features[start] - spatial_features[end])**2, dim=-1) 
        grav_function, grav_fact = self.get_grav_function(d)

        return torch.exp(grav_function), grav_fact

    def grav_pooling(self, spatial_features, hidden_features):
        edge_index = self.get_neighbors(spatial_features)
        start, end = edge_index
        d_weight, grav_fact = self.get_attention_weight(spatial_features, hidden_features, edge_index)

        if "norm_hidden" in self.hparams and self.hparams["norm_hidden"]:
            hidden_features = F.normalize(hidden_features, p=1, dim=-1)

        return scatter_add(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=hidden_features.shape[0]), edge_index, grav_fact

    def forward(self, hidden_features, batch, current_epoch):
        self.current_epoch = current_epoch
        self.batch = batch

        hidden_features = torch.cat([hidden_features, hidden_features.mean(dim=1).unsqueeze(-1)], dim=-1)
        spatial_features = self.spatial_network(hidden_features)

        if "norm_embedding" in self.hparams and self.hparams["norm_embedding"]:
            spatial_features = F.normalize(spatial_features, p=2, dim=-1)

        aggregated_hidden, edge_index, grav_fact = self.grav_pooling(spatial_features, hidden_features)
        concatenated_hidden = torch.cat([aggregated_hidden, hidden_features], dim=-1)
        return self.feature_network(concatenated_hidden), edge_index, spatial_features, grav_fact

    def setup_neighborhood_configuration(self):
        self.current_epoch = 0
        self.use_radius = bool("r" in self.hparams and self.hparams["r"])
        # A fix here for the case where there is dropout and a large embedded space, model initially can't find neighbors: Enforce self-loop
        if not self.hparams["knn"] and self.hparams["emb_dims"] > 4 and (self.hparams["feature_dropout"] or self.hparams["spatial_dropout"]):
            self.hparams["self_loop"] = True
        self.use_knn = bool("knn" in self.hparams and self.hparams["knn"])
        self.use_rand_k = bool("rand_k" in self.hparams and self.hparams["rand_k"])

    @property
    def r(self):
        if isinstance(self.hparams["r"], list):
            if len(self.hparams["r"]) == 2:
                return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / self.hparams["max_epochs"] )
            elif len(self.hparams["r"]) == 3:
                if self.current_epoch < self.hparams["max_epochs"]/2:
                    return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / (self.hparams["max_epochs"]/2) )
                else:
                    return self.hparams["r"][1] + ( (self.hparams["r"][2] - self.hparams["r"][1]) * (self.current_epoch - self.hparams["max_epochs"]/2) / (self.hparams["max_epochs"]/2) )
        elif isinstance(self.hparams["r"], float):
            return self.hparams["r"]
        else:
            return 0.3

    @property
    def knn(self):
        if not isinstance(self.hparams["knn"], list):
            return self.hparams["knn"]
        if len(self.hparams["knn"]) == 2:
            return int( self.hparams["knn"][0] + ( (self.hparams["knn"][1] - self.hparams["knn"][0]) * self.current_epoch / self.hparams["max_epochs"] ) )
        elif len(self.hparams["knn"]) == 3:
            return int(self.hparams["knn"][0] + ((self.hparams["knn"][1] - self.hparams["knn"][0]) * self.current_epoch / (self.hparams["max_epochs"] / 2))) if self.current_epoch < self.hparams["max_epochs"] / 2 else int(self.hparams["knn"][1] + ((self.hparams["knn"][2] - self.hparams["knn"][1]) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"] / 2)))
        else:
            raise ValueError("knn must be a list of length 2 or 3")

    @property
    def rand_k(self):        
        if not isinstance(self.hparams["rand_k"], list):
            return self.hparams["rand_k"]
        if len(self.hparams["knn"]) == 2:
            return int( self.hparams["rand_k"][0] + ( (self.hparams["rand_k"][1] - self.hparams["rand_k"][0]) * self.current_epoch / self.hparams["max_epochs"] ) )
        elif len(self.hparams["rand_k"]) == 3:
            return int(self.hparams["rand_k"][0] + ((self.hparams["rand_k"][1] - self.hparams["rand_k"][0]) * self.current_epoch / (self.hparams["max_epochs"] / 2))) if self.current_epoch < self.hparams["max_epochs"] / 2 else int(self.hparams["rand_k"][1] + ((self.hparams["rand_k"][2] - self.hparams["rand_k"][1]) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"] / 2)))
        else:
            raise ValueError("rand_k must be a list of length 2 or 3")

    @property
    def grav_weight(self):        
        if isinstance(self.hparams["grav_weight"], list) and len(self.hparams["grav_weight"]) == 2:
            return (self.hparams["grav_weight"][0] + (self.hparams["grav_weight"][1] - self.hparams["grav_weight"][0]) * self.current_epoch / self.hparams["max_epochs"])
        elif isinstance(self.hparams["grav_weight"], float):
            return self.hparams["grav_weight"]
        else:
            raise ValueError("grav_weight must be a list of length 2 or a float")
        



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

    def forward(self, x):
        x_in = x
        x = self.gravnet(x)
        x = self.lin(x)
        x = self.norm(x)
        x = self.act(x)
        return x + x_in



class GravNetModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.input_lin = nn.Linear(hparams["node_features"], hparams["hidden_dim"])

        # Stack of GravNet blocks
        self.blocks = nn.ModuleList([
            GravNetBlock(hparams["hidden_dim"], hparams["hidden_dim"]),
            GravNetBlock(hparams["hidden_dim"], hparams["hidden_dim"]),
            GravNetBlock(hparams["hidden_dim"], hparams["hidden_dim"])
        ])

        # Event-level MLP
        self.event_mlp = nn.Sequential(
            nn.Linear(hparams["hidden_dim"], 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # Classification head
        self.cls_head = nn.Linear(128, hparams["n_classes"])
        # Energy regression head
        self.energy_head = nn.Linear(128, 1)

    def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x, batch = data.x, data.batch

        x = self.input_lin(x)
        for block in self.blocks:
            # x = block(x, edge_index, batch)
            # print(x, batch)
            x = block(x)

        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.event_mlp(x)

        class_logits = self.cls_head(x)
        energy_pred = self.energy_head(x).squeeze(-1)
        return class_logits, energy_pred

