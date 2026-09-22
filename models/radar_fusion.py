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
                 mode='current', max_speed=35., age_decay=.5, horizon_decay=3.,
                 velocity_consistency=False, temporal_reliability=False):
        super().__init__()
        if mode not in ('current', 'transport'):
            raise ValueError(mode)
        if age_decay <= 0 or horizon_decay <= 0:
            raise ValueError('Confidence time scales must be positive')
        self.mode = mode
        self.max_speed = max_speed
        self.age_decay = age_decay
        self.horizon_decay = horizon_decay
        self.velocity_consistency = velocity_consistency
        self.temporal_reliability = temporal_reliability
        if (velocity_consistency or temporal_reliability) and mode != 'transport':
            raise ValueError('Reliability extensions require causal transport')
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
        # Added AFTER A's parameters: their seed-0 initialization is unchanged.
        if velocity_consistency:
            # Bounded 1--10 m/s kernel width, initially 3 m/s.
            self.velocity_scale = nn.Parameter(torch.tensor(math.log(2. / 7.)))
        if temporal_reliability:
            with torch.random.fork_rng(devices=[]):
                self.reliability_gate = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.reliability_gate[-1].weight)
            nn.init.zeros_(self.reliability_gate[-1].bias)

    def velocity_agreement(self, neighbors, spatial_weights):
        """Leave-one-out radial-proxy agreement, never compare a return to itself.

        Each peer's compensated vector is projected onto the tested return's
        cached planar LOS. These are SDK-derived proxies, not raw Doppler or
        full object velocities. Singletons retain the spatial association.
        """
        points, weights = neighbors.float(), spatial_weights.float()
        velocity, los, radial = points[..., 3:5], points[..., 8:10], points[..., 7]
        prediction = torch.einsum('...id,...jd->...ij', los, velocity)
        residual = radial[..., :, None] - prediction
        count = neighbors.shape[-2]
        off_diagonal = ~torch.eye(count, device=points.device, dtype=torch.bool)
        peers = weights[..., None, :] * off_diagonal
        sigma = 1. + 9. * self.velocity_scale.float().sigmoid()
        agreement = (torch.exp(-.5 * (residual / sigma).square()) * peers).sum(-1)
        denominator = peers.sum(-1)
        agreement = agreement / denominator.clamp_min(1e-8)
        return torch.where(denominator > 1e-8, agreement, torch.ones_like(agreement))

    def temporal_factor(self, quality, horizon_seconds):
        # Initially exactly A's exp(-t/3); positive bounded learned decay rates
        # prevent amplification and keep current-time fusion unchanged.
        delta = self.reliability_gate(quality).float()
        rate = delta.tanh().exp() / self.horizon_decay
        return torch.exp(-horizon_seconds * rate)

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
        if self.velocity_consistency or self.temporal_reliability:
            neighbors = radar[batch, indices].float()
            agreement = (self.velocity_agreement(neighbors, weights) if self.velocity_consistency
                         else torch.ones_like(weights))
            base_weights = weights.float()
            if self.velocity_consistency:
                # A nonzero floor preserves spatial fallback for ambiguous or
                # multi-object neighborhoods; do not hard-reject disagreements.
                weights = weights * (.25 + .75 * agreement)
        pooled = (neighbor_features * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        sample_valid = mask.any(-1)
        sample_weights = self.sample_weight(query).softmax(-1) * sample_valid
        sample_weights = sample_weights / sample_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        fused = (pooled * sample_weights[..., None]).sum(-2)
        update = self.output(fused) * self.gate(torch.cat([query, fused], -1)).sigmoid()
        if self.mode == 'transport':
            if self.temporal_reliability:
                normalized = base_weights / base_weights.sum(-1, keepdim=True).clamp_min(1e-8)
                mean_velocity = (normalized[..., None] * neighbors[..., 3:5]).sum(-2)
                spread = ((neighbors[..., 3:5] - mean_velocity[..., None, :]).square().sum(-1)
                          * normalized).sum(-1).clamp_min(1e-8).sqrt()
                quality = torch.stack([
                    (normalized * neighbors[..., 6]).sum(-1) / .5,
                    mask.float().mean(-1),
                    (normalized * agreement).sum(-1),
                    (spread / 20.).clamp(max=2.),
                    (normalized * torch.where(mask, distances.float(), 0.)).sum(-1) / self.radius**2,
                    torch.full_like(spread, horizon_seconds / 3.)], -1)
                quality = (quality * sample_weights.float()[..., None]).sum(-2)
                update = update * self.temporal_factor(quality.to(query.dtype), horizon_seconds).to(update.dtype)
            else:
                update = update * math.exp(-horizon_seconds / self.horizon_decay)
        return query + update * sample_valid.any(-1, keepdim=True)
