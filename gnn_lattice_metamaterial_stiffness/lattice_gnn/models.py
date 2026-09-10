# graph models and graph-agnostic anns

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_add_pool, global_mean_pool, global_max_pool,
    MessagePassing,
)

# compute a population standard deviation for every graph and channel
def _global_std_pool(x: torch.Tensor, batch: torch.Tensor, size: int | None = None) -> torch.Tensor:
    mean = global_mean_pool(x, batch, size=size)
    centered = x - mean[batch]
    var = global_mean_pool(centered * centered, batch, size=size)
    return torch.sqrt(var.clamp_min(1e-12))

# mean pool only entities selected by a physical entity mask
def _global_mean_pool_masked(
    x: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor | None,
    size: int | None = None,
) -> torch.Tensor:
    if mask is None:
        return global_mean_pool(x, batch, size=size)
    mask = mask.view(-1).to(device=x.device, dtype=torch.bool)
    if mask.numel() != x.size(0) or bool(mask.all()):
        return global_mean_pool(x, batch, size=size)
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 1
    if not bool(mask.any()):
        return x.new_zeros((int(size), x.size(1)))
    return global_mean_pool(x[mask], batch[mask], size=size)

# validate nonnegative centralities used only by graph-level pooling
def _pooling_centrality(
    value: torch.Tensor | None,
    x: torch.Tensor,
    expected_dim: int,
) -> torch.Tensor:
    if expected_dim == 0:
        return x.new_empty((x.size(0), 0))
    if value is None:
        raise ValueError("centrality-guided pooling weights are missing")
    value = value.to(device=x.device, dtype=x.dtype)
    if value.dim() != 2 or value.shape != (x.size(0), expected_dim):
        raise ValueError(
            "centrality-guided pooling weights have an unexpected shape"
        )
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

# compute one normalized weighted mean per centrality descriptor
def _global_centrality_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    centrality: torch.Tensor,
    size: int | None = None,
) -> torch.Tensor:
    if centrality.size(1) == 0:
        if size is None:
            size = int(batch.max().item()) + 1 if batch.numel() else 1
        return x.new_empty((int(size), 0))
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 1
    ordinary_mean = global_mean_pool(x, batch, size=size)
    num_centralities = int(centrality.size(1))
    hidden_dim = int(x.size(1))
    weighted_values = (
        centrality.unsqueeze(-1) * x.unsqueeze(1)
    ).reshape(x.size(0), num_centralities * hidden_dim)
    numerator = global_add_pool(
        weighted_values,
        batch,
        size=size,
    ).reshape(int(size), num_centralities, hidden_dim)
    denominator = global_add_pool(
        centrality,
        batch,
        size=size,
    ).unsqueeze(-1)
    weighted_mean = numerator / denominator.clamp_min(1e-12)
    pooled = torch.where(
        denominator > 1e-12,
        weighted_mean,
        ordinary_mean.unsqueeze(1),
    )
    return pooled.reshape(int(size), num_centralities * hidden_dim)

# learn a convex mixture of centrality-weighted mean contrasts
# each contrast is a centrality-weighted entity mean minus the ordinary entity mean
# softmax-normalized logits learn the relative contribution of degree, closeness, betweenness, and eigenvector centrality
class CentralityMixturePool(nn.Module):

    def __init__(self, centrality_dim: int):
        super().__init__()
        self.centrality_dim = int(centrality_dim)
        self.logits = nn.Parameter(torch.zeros(self.centrality_dim))

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        centrality: torch.Tensor,
        mask: torch.Tensor | None = None,
        size: int | None = None,
    ) -> torch.Tensor:
        if size is None:
            size = int(batch.max().item()) + 1 if batch.numel() else 1
        if mask is not None:
            mask = mask.view(-1).to(device=x.device, dtype=torch.bool)
            if mask.numel() == x.size(0) and not bool(mask.all()):
                if not bool(mask.any()):
                    return x.new_zeros((int(size), x.size(1)))
                x = x[mask]
                batch = batch[mask]
                centrality = centrality[mask]
        if self.centrality_dim == 0:
            return x.new_zeros((int(size), x.size(1)))
        ordinary_mean = global_mean_pool(x, batch, size=size)
        pooled = _global_centrality_pool(
            x, batch, centrality, size=size
        ).reshape(int(size), self.centrality_dim, x.size(1))
        contrasts = pooled - ordinary_mean.unsqueeze(1)
        mixture = torch.softmax(self.logits, dim=0).view(
            1, self.centrality_dim, 1
        )
        return (contrasts * mixture).sum(dim=1)

# combine ordinary statistics with one learned centrality contrast
def _pool_with_centrality(
    x: torch.Tensor,
    batch: torch.Tensor,
    centrality: torch.Tensor,
    mixture_pool: CentralityMixturePool,
    mask: torch.Tensor | None = None,
    size: int | None = None,
) -> torch.Tensor:
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 1
    centrality_contrast = mixture_pool(
        x, batch, centrality, mask=mask, size=size
    )
    if mask is not None:
        mask = mask.view(-1).to(device=x.device, dtype=torch.bool)
        if mask.numel() == x.size(0) and not bool(mask.all()):
            if not bool(mask.any()):
                return x.new_zeros((int(size), x.size(1) * 4))
            x = x[mask]
            batch = batch[mask]
    ordinary_mean = global_mean_pool(x, batch, size=size)
    return torch.cat(
        [
            ordinary_mean,
            centrality_contrast,
            global_max_pool(x, batch, size=size),
            _global_std_pool(x, batch, size=size),
        ],
        dim=1,
    )

# pool raw features by ordinary mean and learned centrality contrast
def _pool_raw_with_centrality(
    x: torch.Tensor,
    batch: torch.Tensor,
    centrality: torch.Tensor,
    mixture_pool: CentralityMixturePool,
    mask: torch.Tensor | None = None,
    size: int | None = None,
) -> torch.Tensor:
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 1
    centrality_contrast = mixture_pool(
        x, batch, centrality, mask=mask, size=size
    )
    ordinary_mean = _global_mean_pool_masked(
        x, batch, mask, size=size
    )
    return torch.cat([ordinary_mean, centrality_contrast], dim=1)

# shared graph model components
# nonlinear messages followed by mean-plus-sum aggregation and update
class MeanSumMPNNConv(MessagePassing):

    def __init__(self, hidden_dim: int):
        super().__init__(aggr=["mean", "sum"])
        self.hidden_dim = int(hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x_source = x
            x_target = x
        else:
            x_source, x_target = x
        if edge_attr is None:
            edge_attr = x_source.new_zeros(
                (edge_index.size(1), self.hidden_dim)
            )
        if edge_attr.size(-1) != self.hidden_dim:
            raise ValueError("encoded edge features must match hidden_dim")
        aggregated = self.propagate(
            edge_index,
            x=(x_source, x_target),
            edge_attr=edge_attr,
            size=(x_source.size(0), x_target.size(0)),
        )
        return self.update_mlp(torch.cat([x_target, aggregated], dim=-1))

    def message(self, x_j, x_i, edge_attr):
        return self.message_mlp(torch.cat([x_j + edge_attr, x_i], dim=-1))


# update one strut state with the shared incidence message operator
class SymmetricEdgeStateUpdate(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.conv = MeanSumMPNNConv(self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, node_state, edge_index, directed_edge_state):
        if edge_index.size(1) != directed_edge_state.size(0):
            raise ValueError("edge indices and edge states must have equal length")
        if directed_edge_state.size(0) % 2 != 0:
            raise ValueError(
                "homogeneous edge states must contain two entries per strut"
            )
        if directed_edge_state.size(0) == 0:
            return directed_edge_state

        unique_edge_state = 0.5 * (
            directed_edge_state[::2] + directed_edge_state[1::2]
        )
        endpoints = edge_index[:, ::2]
        num_struts = int(unique_edge_state.size(0))
        strut_ids = torch.arange(num_struts, device=edge_index.device)
        incidence_edge_index = torch.stack(
            [
                torch.cat([endpoints[0], endpoints[1]], dim=0),
                torch.cat([strut_ids, strut_ids], dim=0),
            ],
            dim=0,
        )
        edge_update = self.conv(
            (node_state, unique_edge_state),
            incidence_edge_index,
        )
        updated_unique_edge_state = F.relu(
            self.norm(unique_edge_state + edge_update)
        )
        return updated_unique_edge_state.repeat_interleave(2, dim=0)


# independent residual node update used by the node-encoder baseline
class ResidualNodeMLP(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return F.relu(self.norm(x + self.mlp(x)))


# bipartite model
# update explicit strut nodes and joint nodes on the incidence graph
# one block first updates every strut from its endpoint joints and then uses the updated strut states to update their incident joints
# separate message operators are learned for the two directions
class BipartiteGNN(nn.Module):
    def __init__(self, in_dim_strut, in_dim_joint, u_dim, hidden_dim, num_layers,
                 include_graph_counts=False, centrality_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.centrality_dim = int(centrality_dim)
        self.include_graph_counts = bool(include_graph_counts)
        self.strut_pool = CentralityMixturePool(self.centrality_dim)
        self.joint_pool = CentralityMixturePool(self.centrality_dim)
        self.lin_strut = nn.Linear(in_dim_strut, hidden_dim)
        self.lin_joint = nn.Linear(in_dim_joint, hidden_dim)
        self.convs = nn.ModuleList([
            nn.ModuleDict({
                "strut_to_joint": MeanSumMPNNConv(hidden_dim),
                "joint_to_strut": MeanSumMPNNConv(hidden_dim),
            })
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                "strut": nn.LayerNorm(hidden_dim),
                "joint": nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])
        count_dim = 2 if self.include_graph_counts else 0
        pooled_dim = hidden_dim * 8
        self.fcl = nn.Linear(pooled_dim + u_dim + count_dim, hidden_dim)
        self.fch = nn.Linear(hidden_dim, hidden_dim//2)
        self.out = nn.Linear(hidden_dim//2, 6)

    def forward(self, data):
        h_s = F.relu(self.lin_strut(data["strut"].x))
        h_j = F.relu(self.lin_joint(data["joint"].x))
        h = {"strut": h_s, "joint": h_j}
        for layer_idx, conv in enumerate(self.convs):
            strut_update = conv["joint_to_strut"](
                (h["joint"], h["strut"]),
                data[("joint", "has", "strut")].edge_index,
            )
            next_strut = F.relu(self.norms[layer_idx]["strut"](
                h["strut"] + strut_update
            ))
            joint_update = conv["strut_to_joint"](
                (next_strut, h["joint"]),
                data[("strut", "incident", "joint")].edge_index,
            )
            next_joint = F.relu(self.norms[layer_idx]["joint"](
                h["joint"] + joint_update
            ))
            h = {"strut": next_strut, "joint": next_joint}
        size = int(data.u.size(0))
        strut_centrality = _pooling_centrality(
            getattr(data["strut"], "pooling_centrality", None),
            h["strut"],
            self.centrality_dim,
        )
        joint_centrality = _pooling_centrality(
            getattr(data["joint"], "pooling_centrality", None),
            h["joint"],
            self.centrality_dim,
        )
        ps = _pool_with_centrality(
            h["strut"],
            data["strut"].batch,
            strut_centrality,
            self.strut_pool,
            size=size,
        )
        pj = _pool_with_centrality(
            h["joint"],
            data["joint"].batch,
            joint_centrality,
            self.joint_pool,
            mask=getattr(data["joint"], "physical_mask", None),
            size=size,
        )
        u  = data.u
        parts = [ps, pj]
        if u.size(1) > 0:
            parts.append(u)
        if self.include_graph_counts:
            parts.append(data.graph_counts)
        x = torch.cat(parts, dim=1)
        x = F.relu(self.fcl(x))
        x = F.relu(self.fch(x))
        return self.out(x)

# update joint nodes while keeping encoded strut edge states fixed
class StaticEdgeHomogeneousGNN(nn.Module):
    def __init__(self, in_dim_joint, in_dim_strut, u_dim, hidden_dim, num_layers,
                 include_graph_counts=False, centrality_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.centrality_dim = int(centrality_dim)
        self.include_graph_counts = bool(include_graph_counts)
        self.strut_pool = CentralityMixturePool(self.centrality_dim)
        self.joint_pool = CentralityMixturePool(self.centrality_dim)
        self.lin_joint = nn.Linear(in_dim_joint, hidden_dim)
        self.lin_strut = nn.Linear(in_dim_strut, hidden_dim)
        self.convs = nn.ModuleList([
            MeanSumMPNNConv(hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        count_dim = 2 if self.include_graph_counts else 0
        pooled_dim = hidden_dim * 8
        self.fcl = nn.Linear(pooled_dim + u_dim + count_dim, hidden_dim)
        self.fch = nn.Linear(hidden_dim, hidden_dim//2)
        self.out = nn.Linear(hidden_dim//2, 6)

    def forward(self, data):
        h = F.relu(self.lin_joint(data.x))
        edge_state = F.relu(self.lin_strut(data.edge_attr))
        for layer_idx, conv in enumerate(self.convs):
            update = conv(h, data.edge_index, edge_state)
            h = F.relu(self.norms[layer_idx](h + update))
        return self._readout(data, h, edge_state)

    def _readout(self, data, h, edge_state):
        b = data.batch if hasattr(data, 'batch') else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        size = int(data.u.size(0))
        joint_centrality = _pooling_centrality(
            getattr(data, "joint_pooling_centrality", None),
            h,
            self.centrality_dim,
        )
        p_j = _pool_with_centrality(
            h,
            b,
            joint_centrality,
            self.joint_pool,
            mask=getattr(data, "physical_mask", None),
            size=size,
        )
        if edge_state.size(0) % 2 != 0:
            raise ValueError("homogeneous edge features must contain two entries per strut")
        strut_state = edge_state[::2]
        strut_batch = b[data.edge_index[0, ::2]]
        strut_centrality = _pooling_centrality(
            getattr(data, "strut_pooling_centrality", None),
            strut_state,
            self.centrality_dim,
        )
        p_s = _pool_with_centrality(
            strut_state,
            strut_batch,
            strut_centrality,
            self.strut_pool,
            size=size,
        )
        parts = [p_s, p_j]
        if data.u.size(1) > 0:
            parts.append(data.u)
        if self.include_graph_counts:
            parts.append(data.graph_counts)
        x = torch.cat(parts, dim=1)
        x = F.relu(self.fcl(x))
        x = F.relu(self.fch(x))
        return self.out(x)


# extend the homogeneous model with one evolving state per strut
# each block updates the undirected strut state from both endpoints before using that updated state in the joint update
# the paired directed edges share the same strut state
class DynamicEdgeHomogeneousGNN(StaticEdgeHomogeneousGNN):
    def __init__(self, in_dim_joint, in_dim_strut, u_dim, hidden_dim, num_layers,
                 include_graph_counts=False, centrality_dim=0):
        super().__init__(
            in_dim_joint,
            in_dim_strut,
            u_dim,
            hidden_dim,
            num_layers,
            include_graph_counts=include_graph_counts,
            centrality_dim=centrality_dim,
        )
        self.edge_updates = nn.ModuleList([
            SymmetricEdgeStateUpdate(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        h = F.relu(self.lin_joint(data.x))
        edge_state = F.relu(self.lin_strut(data.edge_attr))
        for layer_idx, (edge_update, conv) in enumerate(
            zip(self.edge_updates, self.convs)
        ):
            next_edge_state = edge_update(h, data.edge_index, edge_state)
            update = conv(h, data.edge_index, next_edge_state)
            next_h = F.relu(self.norms[layer_idx](h + update))
            edge_state = next_edge_state
            h = next_h
        return self._readout(data, h, edge_state)


# ann baselines
# transform joints and struts independently without graph connectivity
class NodeEncoderANN(nn.Module):
    def __init__(self, in_dim_strut, in_dim_joint, u_dim, hidden_dim, num_layers,
                 include_graph_counts=False, centrality_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.centrality_dim = int(centrality_dim)
        self.include_graph_counts = bool(include_graph_counts)
        self.strut_pool = CentralityMixturePool(self.centrality_dim)
        self.joint_pool = CentralityMixturePool(self.centrality_dim)
        self.lin_s = nn.Linear(in_dim_strut, hidden_dim)
        self.lin_j = nn.Linear(in_dim_joint, hidden_dim)
        self.supd = nn.ModuleList([
            ResidualNodeMLP(hidden_dim) for _ in range(num_layers)
        ])
        self.jupd = nn.ModuleList([
            ResidualNodeMLP(hidden_dim) for _ in range(num_layers)
        ])
        self.read  = nn.Sequential(
            nn.Linear(
                hidden_dim * 8
                + u_dim + (2 if self.include_graph_counts else 0),
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 6),
        )

    def forward(self, data):
        x_s, b_s = data["strut"].x, data["strut"].batch
        x_j, b_j = data["joint"].x, data["joint"].batch
        h_s = F.relu(self.lin_s(x_s))
        h_j = F.relu(self.lin_j(x_j)) if x_j.numel() else torch.zeros((0,self.lin_j.out_features), device=x_s.device)
        for sup, jup in zip(self.supd, self.jupd):
            h_s = sup(h_s)
            if h_j.numel(): h_j = jup(h_j)
        size = int(data.u.size(0))
        strut_centrality = _pooling_centrality(
            getattr(data["strut"], "pooling_centrality", None),
            h_s,
            self.centrality_dim,
        )
        p_s = _pool_with_centrality(
            h_s,
            b_s,
            strut_centrality,
            self.strut_pool,
            size=size,
        )
        if h_j.numel():
            joint_centrality = _pooling_centrality(
                getattr(data["joint"], "pooling_centrality", None),
                h_j,
                self.centrality_dim,
            )
            p_j = _pool_with_centrality(
                h_j,
                b_j,
                joint_centrality,
                self.joint_pool,
                mask=getattr(data["joint"], "physical_mask", None),
                size=size,
            )
        else:
            p_j = p_s.new_zeros(p_s.shape)
        u = data.u
        parts = [p_s, p_j]
        if u.size(1) > 0:
            parts.append(u)
        if self.include_graph_counts:
            parts.append(data.graph_counts)
        g = torch.cat(parts, dim=1)
        g = self.read[1](self.read[0](g))
        g = self.read[3](self.read[2](g))
        return self.read[4](g)

# pool raw joint and strut features without learned entity encoders
class RawFeatureMeanPoolingANN(nn.Module):
    def __init__(self, strut_dim, joint_dim, u_dim, hidden_dim,
                 include_graph_counts=False, centrality_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.centrality_dim = int(centrality_dim)
        self.strut_dim = int(strut_dim)
        self.joint_dim = int(joint_dim)
        self.include_graph_counts = bool(include_graph_counts)
        self.strut_pool = CentralityMixturePool(self.centrality_dim)
        self.joint_pool = CentralityMixturePool(self.centrality_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                2 * (strut_dim + joint_dim) + u_dim
                + (2 if self.include_graph_counts else 0),
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 6),
        )

    def forward(self, data):
        xs, bs = data["strut"].x, data["strut"].batch
        xj, bj = data["joint"].x, data["joint"].batch
        size = int(data.u.size(0))
        strut_centrality = _pooling_centrality(
            getattr(data["strut"], "pooling_centrality", None),
            xs,
            self.centrality_dim,
        )
        p_s = _pool_raw_with_centrality(
            xs,
            bs,
            strut_centrality,
            self.strut_pool,
            size=size,
        )
        p_j = (
            _pool_raw_with_centrality(
                xj,
                bj,
                _pooling_centrality(
                    getattr(data["joint"], "pooling_centrality", None),
                    xj,
                    self.centrality_dim,
                ),
                self.joint_pool,
                mask=getattr(data["joint"], "physical_mask", None),
                size=size,
            )
            if xj.numel()
            else p_s.new_zeros((p_s.size(0), self.joint_dim * 2))
        )
        u   = data.u
        parts = [p_s, p_j]
        if u.size(1) > 0:
            parts.append(u)
        if self.include_graph_counts:
            parts.append(data.graph_counts)
        z = torch.cat(parts, dim=1)
        z = self.mlp[1](self.mlp[0](z))
        z = self.mlp[3](self.mlp[2](z))
        return self.mlp[4](z)
