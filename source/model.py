import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from torch_geometric.nn import GravNetConv, global_mean_pool, global_max_pool
from torch_geometric.nn import global_mean_pool, global_max_pool
import math

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
from torch_geometric.nn import pool
from torch_geometric.nn import GraphNorm


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

import torch
import torch.nn as nn

class LearnedDownsampling(nn.Module):
    def __init__(self, in_channels, ratio=0.5, min_nodes=100):
        """
        Args:
            in_channels: node feature dimension
            ratio: fraction of nodes to keep per graph
            min_nodes: minimum nodes kept per graph
        """
        super().__init__()
        self.score = nn.Linear(in_channels, 1)
        self.ratio = ratio
        self.min_nodes = min_nodes

    def forward(self, x, batch):
        """
        x: [N, F]
        batch: [N]
        """
        scores = self.score(x).squeeze(-1)

        unique_batches = batch.unique()
        perm_list = []

        for b in unique_batches:
            mask = batch == b
            idx = mask.nonzero(as_tuple=False).view(-1)

            xb = x[idx]
            sb = scores[idx]

            k = max(self.min_nodes, int(self.ratio * xb.size(0)))
            k = min(k, xb.size(0))

            topk = torch.topk(sb, k, sorted=False).indices
            perm_list.append(idx[topk])

        perm = torch.cat(perm_list, dim=0)

        x = x[perm]
        batch = batch[perm]

        return x, batch, perm


class FastGravNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Hit embedding
        self.embed = nn.Linear(cfg.input_dim, cfg.embed_dim)
        
        # GravNet layers
        self.grav_convs = nn.ModuleList()
        for i in range(len(cfg.grav_conv_dims)):
            in_dim = cfg.embed_dim
            out_dim = cfg.grav_conv_dims[i]
            if i > 0:
                in_dim = cfg.grav_conv_dims[i-1]
                out_dim = cfg.grav_conv_dims[i]
            self.grav_convs.append(GravNetConv(in_dim, 
                                               out_dim, 
                                               space_dimensions=cfg.space_dims[i], 
                                               k=cfg.knn[i], 
                                               propagate_dimensions=cfg.propagate_dims[i]))

        # Learned Downsampling Layer
        # self.down_sample = LearnedDownsampling(cfg.embed_dim, 
        #                                        ratio=cfg.downsample_frac, 
        #                                        min_nodes=cfg.downsample_min_hits)
        
        # self.down_sample = pool.TopKPooling(cfg.embed_dim, ratio=cfg.downsample_frac, min_score=0.0)
        
        # Batch norm layers
        self.conv_batch_norms = nn.ModuleList()
        self.bn0 = nn.BatchNorm1d(cfg.embed_dim, momentum=0.05)
        for dim in cfg.grav_conv_dims:
            # self.conv_batch_norms.append(nn.BatchNorm1d(dim, momentum=0.05))
            self.conv_batch_norms.append(GraphNorm(dim))

        # Dropout 
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.act = nn.ReLU()
        
        # Classifier Head
        if cfg.do_classification:
            self.classifier = nn.Sequential(
                nn.Linear(cfg.grav_conv_dims[-1], 32),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(16, cfg.nclasses),
            )
        
        # Regression Head
        if cfg.do_regression:
            # self.regressor_out = nn.Linear(16, cfg.n_targets)
            # self.regressor = nn.Sequential(
            #     nn.Linear(cfg.grav_conv_dims[-1], 32),
            #     nn.ReLU(),
            #     nn.Dropout(cfg.dropout),
            #     nn.Linear(32, 16),
            #     nn.ReLU(),
            #     nn.Dropout(cfg.dropout),
            #     nn.Linear(16, cfg.n_targets),
            # )

            self.regressor = nn.Sequential(
                nn.Linear(cfg.grav_conv_dims[-1], 32),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.BatchNorm1d(32, momentum=0.05),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.BatchNorm1d(16, momentum=0.05),
                nn.Linear(16, cfg.n_targets),
                # nn.Sigmoid(),  # constrain 0-1
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.embed(x)
        x = self.dropout1(x)

        # x, edge_index, _, batch, _, _ = self.down_sample(
        #     x, edge_index, None, batch
        # )

        x = self.bn0(x)

        for grav_conv, bn in zip(self.grav_convs, self.conv_batch_norms):
            x_in = x
            x = grav_conv(x, batch=batch)
            x = bn(x)
            x = self.act(x)
            x = x + x_in  # residual



        x = global_mean_pool(x, batch)

        class_head, regr_head = None, None
        if self.cfg.do_classification:
            class_head = self.classifier(x)
        if self.cfg.do_regression:
            
            regr_head = self.regressor(x)
            # regr_head = self.regressor_out(x)
            # logE_min = 0.0
            # logE_max = math.log10(4500)
            # mid = (logE_max + logE_min) / 2
            # half_range = (logE_max - logE_min) / 2
            # regr_head = torch.tanh(regr_head) * half_range + mid
            
            # if regr_head.shape[-1] == 1:
            #     regr_head = regr_head.squeeze(-1)

        return class_head, regr_head



class NeutrinoGravNetWithRegression(nn.Module):
    """
    GravNet model for both graph-level neutrino event classification
    and node-level hit classification.

    Based on https://arxiv.org/pdf/1902.07987.pdf

    Architecture:
    - GlobalExchange: append global mean to each node
    - 3 GravNet blocks with feature transformation MLPs
    - Node-level classification head for hit identification
    - Global pooling for graph-level prediction
    - Graph-level classification head for event classification
    """

    def __init__(
        self,
        cfg,
        input_dim: int = 0,
        num_graph_classes: int = 3,
        num_node_classes: int = 3,
        dropout: float = 0.2,
        n_feature_transform: int = 16,
        out_channels: int = 16,
        space_dimensions: int = 3,
        propagate_dimensions: int = 16,
        k: int = 16,
        n_gravstack: int = 3,
        batchnorm_momentum: float = 0.05,
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., energy only)
            num_graph_classes: Number of graph-level classes (event classification)
            num_node_classes: Number of node-level classes (hit classification)
            dropout: Dropout probability
            n_feature_transform: Hidden dimension for feature transformation MLPs
            out_channels: Output channels from each GravNet block
            space_dimensions: Dimensionality of learned spatial representation (S)
            propagate_dimensions: Dimensionality of features to propagate (F_LR)
            k: Number of nearest neighbors for aggregation
            n_gravstack: Number of GravNet blocks
            batchnorm_momentum: BatchNorm momentum (use 1-momentum from TF convention)
        """
        super().__init__()

        self.cfg = cfg

        # Input will be [features, x, y, z] concatenated
        input_with_pos = input_dim + 3

        # After GlobalExchange, input doubles (original + global mean)
        initial_features = input_with_pos * 2

        # GravNet stack 1
        self.ft1_1 = nn.Linear(initial_features, n_feature_transform)
        self.ft1_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft1_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn1 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 2
        self.ft2_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft2_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft2_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn2 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # GravNet stack 3
        self.ft3_1 = nn.Linear(out_channels, n_feature_transform)
        self.ft3_2 = nn.Linear(n_feature_transform, n_feature_transform)
        self.ft3_3 = nn.Linear(n_feature_transform, n_feature_transform)
        self.gn3 = GravNetConv(
            in_channels=n_feature_transform,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=batchnorm_momentum)

        # After concatenating all GravNet outputs
        concat_features = n_gravstack * out_channels

        # Graph-level classification head
        self.graph_pooling = global_mean_pool

        self.graph_classifier = nn.Sequential(
            nn.Linear(concat_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, cfg.n_targets),
        )

    def forward(self, data):
        """
        Forward pass for both node-level and graph-level classification.

        Args:
            x: Node features [N, input_dim] (e.g., energy)
            pos: Node positions [N, 3] (x, y, z coordinates)
            batch: Batch assignment vector [N] for batched graphs

        Returns:
            node_out: Node predictions [N, num_node_classes]
            graph_out: Graph predictions [batch_size, num_graph_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Create batch tensor if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Concatenate features and positions
        # x = torch.cat([x, pos], dim=-1)  # [N, input_dim + 3]

        # GlobalExchange: append mean of all features to each node
        # This provides global context to each node
        global_mean = global_mean_pool(x, batch)  # [batch_size, input_dim + 3]
        x = torch.cat([x, global_mean[batch]], dim=-1)  # [N, 2*(input_dim + 3)]

        # List to hold outputs from each GravNet block
        feat = []

        # GravNet stack 1
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x))
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)

        # GravNet stack 2
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x))
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)

        # GravNet stack 3
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)

        # Concatenate all GravNet block outputs
        x = torch.cat(feat, dim=1)  # [N, n_gravstack * out_channels]


        # Global pooling for graph-level prediction
        x_pooled = self.graph_pooling(
            x, batch
        )  # [batch_size, n_gravstack * out_channels]

        # Graph classification
        graph_out = self.graph_classifier(x_pooled)

        return None, graph_out
