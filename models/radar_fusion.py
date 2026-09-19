"""M0: current-query radar association, deformable samples and residual gating.

Inspired by Sparse4D-Radar, not its unpublished implementation. nuScenes radar
has no reliable elevation: association uses XY, while Z remains an input feature.
The optional transport variant exposes causally extrapolated radar evidence to
future queries. The default retains the original M0 current-only behavior.
"""
import math
import torch
from torch import nn


def advect_radar(radar, seconds, max_speed=35.):
    """Constant-velocity proposal in the fixed current-ego coordinate frame.

    Radar positions are measured at t=-age. Ego compensation rotates velocity
    but does not move an object's historical return to the reference timestamp.
    Only past/current observations and the requested horizon are used here.
    """
    if seconds < 0:
        raise ValueError('Forecast horizon must be nonnegative')
    moved = radar.clone()
    moved[..., :2] = radar[..., :2] + radar[..., 3:5] * (radar[..., 6:7] + seconds)
    reliable = (torch.isfinite(radar).all(-1) & (radar[..., 6] >= 0) &
                (radar[..., 6] <= .5) & (radar[..., 3:5].float().norm(dim=-1) <= max_speed))
    moved = torch.where(reliable[..., None], moved, torch.zeros_like(moved))
    return moved, reliable


def centers_to_current(centers, future_to_current):
    """Transform future-ego query centers; leave radar velocity in current ego."""
    matrix = future_to_current.to(device=centers.device, dtype=centers.dtype)
    return torch.einsum('bij,bqj->bqi', matrix[:, :3, :3], centers) + matrix[:, None, :3, 3]


class RadarQueryFusion(nn.Module):
    def __init__(self, embed_dims=256, input_dims=10, num_samples=4,
                 neighbors=8, radius=4.0, max_offset=2.0, gate_bias=-2.0,
                 mode='current', max_speed=35., age_decay=.5, horizon_decay=3.):
        super().__init__()
        if mode not in ('current', 'transport'):
            raise ValueError(mode)
        if age_decay <= 0 or horizon_decay <= 0:
            raise ValueError('Confidence time scales must be positive')
        self.mode = mode
        self.max_speed = max_speed
        self.age_decay = age_decay
        self.horizon_decay = horizon_decay
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

    def forward(self, query, centers, radar, valid, horizon_seconds=0.):
        """query B,Q,C; centers B,Q,3 metres; radar B,R,10; valid B,R.

        Features: x,y,z,vx_comp,vy_comp,rcs,age,radial_comp,los_x,los_y.
        Padding and neighborhoods without returns contribute exactly zero.
        """
        B, Q, C = query.shape
        if radar.shape[1] == 0:
            # Also keep parameters in the graph for empty-radar DDP batches.
            return query + sum(p.sum() * 0 for p in self.parameters())
        if self.mode == 'transport':
            radar, reliable = advect_radar(radar, horizon_seconds, self.max_speed)
            valid = valid & reliable
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
        if self.mode == 'transport':
            weights = weights * torch.exp(-radar[batch, indices, 6] / self.age_decay)
        pooled = (neighbor_features * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        sample_valid = mask.any(-1)
        sample_weights = self.sample_weight(query).softmax(-1) * sample_valid
        sample_weights = sample_weights / sample_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        fused = (pooled * sample_weights[..., None]).sum(-2)
        update = self.output(fused) * self.gate(torch.cat([query, fused], -1)).sigmoid()
        if self.mode == 'transport':
            update = update * math.exp(-horizon_seconds / self.horizon_decay)
        return query + update * sample_valid.any(-1, keepdim=True)
