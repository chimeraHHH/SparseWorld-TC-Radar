"""M0: current-query radar association, deformable samples and residual gating.

Inspired by Sparse4D-Radar, not its unpublished implementation. nuScenes radar
has no reliable elevation: association uses XY, while Z remains an input feature.
There is deliberately no future motion propagation or velocity-consistency head.
"""
import torch
from torch import nn


class RadarQueryFusion(nn.Module):
    def __init__(self, embed_dims=256, input_dims=10, num_samples=4,
                 neighbors=8, radius=4.0, max_offset=2.0, gate_bias=-2.0):
        super().__init__()
        self.num_samples = num_samples
        self.neighbors = neighbors
        self.radius = radius
        self.max_offset = max_offset
        self.point_encoder = nn.Sequential(nn.Linear(input_dims, embed_dims),
            nn.LayerNorm(embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims))
        self.offset = nn.Linear(embed_dims, num_samples * 2)
        self.sample_weight = nn.Linear(embed_dims, num_samples)
        self.relative_encoder = nn.Sequential(nn.Linear(2, embed_dims), nn.ReLU(),
                                               nn.Linear(embed_dims, embed_dims))
        self.gate = nn.Linear(embed_dims * 2, embed_dims)
        self.output = nn.Linear(embed_dims, embed_dims, bias=False)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        # Distinct initial samples preserve learnable spatial coverage.
        with torch.no_grad():
            angle = torch.arange(num_samples) * (2 * torch.pi / num_samples)
            self.offset.bias.copy_(torch.stack([angle.cos(), angle.sin()], -1).flatten() * 0.25)
        nn.init.zeros_(self.sample_weight.weight)
        nn.init.zeros_(self.sample_weight.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, query, centers, radar, valid):
        """query B,Q,C; centers B,Q,3 metres; radar B,R,10; valid B,R.

        Features: x,y,z,vx_comp,vy_comp,rcs,age,radial_comp,los_x,los_y.
        Padding and neighborhoods without returns contribute exactly zero.
        """
        B, Q, C = query.shape
        if radar.shape[1] == 0:
            # Also keep parameters in the graph for empty-radar DDP batches.
            return query + sum(p.sum() * 0 for p in self.parameters())
        scale = radar.new_tensor([40., 40., 5., 20., 20., 20., 1., 20., 1., 1.])
        features = self.point_encoder(radar / scale)
        samples = centers[..., None, :2] + self.offset(query).reshape(B, Q, self.num_samples, 2).tanh() * self.max_offset
        delta = radar[:, None, None, :, :2] - samples[..., None, :]
        dist2 = delta.square().sum(-1).masked_fill(~valid[:, None, None, :], float('inf'))
        distances, indices = dist2.topk(min(self.neighbors, radar.shape[1]), largest=False, dim=-1)
        batch = torch.arange(B, device=query.device)[:, None, None, None]
        neighbor_features = features[batch, indices]
        relative = radar[batch, indices, :2] - samples[..., None, :]
        neighbor_features = neighbor_features + self.relative_encoder(relative / self.radius)
        mask = (distances <= self.radius ** 2) & valid[batch, indices]
        # Unnormalized Gaussian followed by explicit normalization avoids all-masked softmax NaNs.
        weights = torch.exp(-distances / (self.radius ** 2 / 2)) * mask
        pooled = (neighbor_features * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        sample_valid = mask.any(-1)
        sample_weights = self.sample_weight(query).softmax(-1) * sample_valid
        sample_weights = sample_weights / sample_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        fused = (pooled * sample_weights[..., None]).sum(-2)
        update = self.output(fused) * self.gate(torch.cat([query, fused], -1)).sigmoid()
        return query + update * sample_valid.any(-1, keepdim=True)
