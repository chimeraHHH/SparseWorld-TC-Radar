"""Shared, bounded scene paths and censored whole-track matching.

The training objective below is an energy marginalization surrogate. Chamfer
and classification costs are not a normalized observation likelihood. Hidden
points and intermediate poses are never invented by this module.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SharedCensoredPath(nn.Module):
    """Predict K whole-scene paths from one horizon-major feature forward.

    A query has two shared 3D coefficients per mode. In current-ego meters its
    displacement at normalized time u is ``max_residual*u*tanh(a + u*b)``.
    Each coordinate is bounded by max_residual over [0, horizon_seconds].
    Only the displacement is rotated to future ego coordinates; official
    normalized points never undergo a coordinate round trip.
    """

    def __init__(self, embed_dims=256, pc_range=(-40., -40., -1., 40., 40., 5.4),
                 num_modes=2, hidden_dims=128, max_residual=6.,
                 horizon_seconds=3., frame_seconds=.5):
        super().__init__()
        if num_modes < 2 or hidden_dims < 1:
            raise ValueError('At least two modes and a positive hidden size are required')
        if len(pc_range) != 6:
            raise ValueError('pc_range must contain three minima and three maxima')
        extent = torch.tensor(pc_range[3:], dtype=torch.float32) - torch.tensor(pc_range[:3], dtype=torch.float32)
        if not torch.isfinite(extent).all() or not (extent > 0).all():
            raise ValueError('pc_range extents must be finite and positive')
        if not (max_residual > 0 and horizon_seconds > 0 and frame_seconds > 0):
            raise ValueError('Residual bound and time scales must be positive')
        self.embed_dims = embed_dims
        self.num_modes = num_modes
        self.max_residual = float(max_residual)
        self.horizon_seconds = float(horizon_seconds)
        self.frame_seconds = float(frame_seconds)
        # Configuration-derived, not a new checkpoint tensor to reconcile.
        self.register_buffer('scene_extent', extent, persistent=False)
        self.path_encoder = nn.ModuleDict({
            'norm': nn.LayerNorm(embed_dims),
            'projection': nn.Linear(embed_dims, hidden_dims),
            'mode_embedding': nn.Embedding(num_modes, hidden_dims),
            'network': nn.Sequential(nn.SiLU(), nn.Linear(hidden_dims, hidden_dims), nn.SiLU()),
        })
        self.path_head = nn.Linear(hidden_dims, 6)
        self.mixture_head = nn.Sequential(nn.LayerNorm(embed_dims),
                                          nn.Linear(embed_dims, hidden_dims), nn.SiLU(),
                                          nn.Linear(hidden_dims, num_modes))
        nn.init.normal_(self.path_encoder['mode_embedding'].weight, std=.1)
        self.zero_residual_outputs()

    def zero_residual_outputs(self):
        """Call after external framework init, preserving hidden mode asymmetry."""
        nn.init.zeros_(self.path_head.weight)
        nn.init.zeros_(self.path_head.bias)
        nn.init.zeros_(self.mixture_head[-1].weight)
        nn.init.zeros_(self.mixture_head[-1].bias)

    def forward(self, query_features, base_points, base_scores, fut2cur, fut_list):
        """Return mode points/scores [K, F*B, Q, P, D] and logits [B,K].

        Features and base outputs are horizon-major. fut2cur[f] maps future
        ego column vectors to current ego, and fut_list[f] contains frame
        indices (0,2,4,6 for the standard half-second dataset cadence).
        There is no ground-truth argument and no test-time branch oracle.
        """
        if query_features.ndim != 3 or base_points.ndim != 4 or base_scores.ndim != 4:
            raise ValueError('Expected features [F*B,Q,C], points/scores [F*B,Q,P,D]')
        count, queries, channels = query_features.shape
        horizons = len(fut2cur)
        if horizons < 2 or count % horizons or len(fut_list) != horizons:
            raise ValueError('Invalid horizon-major batch or horizon metadata')
        batch = count // horizons
        if channels != self.embed_dims or base_points.shape[:2] != (count, queries):
            raise ValueError('Feature and point dimensions disagree')
        if base_points.shape[-1] != 3 or base_scores.shape[:-1] != base_points.shape[:-1]:
            raise ValueError('Point and semantic dimensions disagree')
        device = base_points.device
        times = []
        rotations = []
        for frame, transform in zip(fut_list, fut2cur):
            frame = torch.as_tensor(frame, device=device, dtype=torch.float32).reshape(-1)
            if frame.numel() == 1:
                frame = frame.expand(batch)
            if frame.numel() != batch:
                raise ValueError('Each horizon must provide one frame index per scene')
            matrix = torch.as_tensor(transform, device=device, dtype=torch.float32)
            if matrix.shape != (batch, 4, 4):
                raise ValueError('Each fut2cur transform must have shape [B,4,4]')
            if not torch.isfinite(matrix).all():
                raise ValueError('Each fut2cur transform must be finite')
            times.append(frame * self.frame_seconds / self.horizon_seconds)
            rotations.append(matrix[:, :3, :3])
        times = torch.stack(times)
        if not torch.isfinite(times).all() or not torch.all(times[0] == 0):
            raise ValueError('The first horizon must be the finite current frame')
        if (times < 0).any() or (times > 1 + 1e-6).any():
            raise ValueError('Forecast times exceed the configured bounded time interval')
        if not torch.all(times[1:] > 0):
            raise ValueError('All subsequent horizons must be future frames')

        # Future-query features contain only observed sensor inputs, never GT.
        features = query_features.reshape(horizons, batch, queries, channels).mean(0)
        dtype = self.path_encoder['projection'].weight.dtype
        encoded = self.path_encoder['projection'](self.path_encoder['norm'](features.to(dtype)))
        mode_embed = self.path_encoder['mode_embedding'].weight[:, None, None, :]
        encoded = self.path_encoder['network'](encoded.unsqueeze(0) + mode_embed)
        coefficients = self.path_head(encoded).float().reshape(self.num_modes, batch, queries, 2, 3)
        logits = self.mixture_head(features.mean(1).to(dtype)).float()
        mode_frames, displacements = [], []
        for horizon in range(horizons):
            base = base_points[horizon * batch:(horizon + 1) * batch]
            if horizon == 0:
                # Direct bypass: even nonzero residual weights cannot change t0.
                mode_frames.append(base.unsqueeze(0).expand(self.num_modes, *base.shape))
                continue
            u = times[horizon][None, :, None, None]
            delta_current = self.max_residual * u * torch.tanh(
                coefficients[..., 0, :] + u * coefficients[..., 1, :])
            # Row vectors: inverse of future->current rotation is delta @ R.
            delta_future = torch.einsum('kbqc,bcd->kbqd', delta_current, rotations[horizon])
            normalized = delta_future / self.scene_extent.to(device=device, dtype=torch.float32)
            mode_frames.append(base.unsqueeze(0) + normalized[..., None, :].to(base.dtype))
            displacements.append(delta_current)
        points = torch.cat(mode_frames, dim=1)
        offsets = torch.stack(displacements, dim=1)
        return {
            'mode_points': points,
            'mode_scores': base_scores.unsqueeze(0).expand(self.num_modes, *base_scores.shape),
            'mode_logits': logits,
            'mode_probability': logits.softmax(-1),
            'mean_offset_m': offsets.norm(dim=-1).mean(dim=(1, 2, 3)),
            'mode_separation_m': (offsets[0] - offsets[1]).norm(dim=-1).mean(),
        }


def censored_sequence_energy(mode_logits, energies, tau=.25):
    """Marginalize one mode over a whole observed sequence, not each endpoint.

    Inputs have identical [..., K] shape. A normalized log-softmax prevents a
    spurious -tau*log(K) reward for duplicated modes. This is an energy
    surrogate, not a claim of a calibrated or normalized likelihood.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    if mode_logits.shape != energies.shape or energies.ndim < 1 or not energies.shape[-1]:
        raise ValueError('Logits and whole-sequence energies must have the same [...,K] shape')
    logits, costs = mode_logits.float(), energies.float()
    return -tau * torch.logsumexp(F.log_softmax(logits, dim=-1) - costs / tau, dim=-1)


def tracker_matching_energy(mode_points, mode_scores, targets, max_tracks=32,
                            max_pred_points=16, geometry_weight=1., semantic_weight=.25,
                            distance_scale=1., return_details=False):
    """Match real observed tracks to queries once over their complete sequence.

    mode_points: decoded current-ego meters [K,F,Q,P,3]. mode_scores: raw
    sigmoid classifier logits [K,F,Q,P,C]. targets contains points [M,F,S,3],
    point_valid [M,F,S], observed [M,F], and semantic labels [M]. Identity is
    supplied by a real annotation track; this function never infers identity
    from semantic class or invents an unobserved endpoint. At least two
    observed frames, including a future frame, are required. t0 need not be
    observed. Upstream must apply its declared deterministic track cap.

    Per-endpoint geometry is a symmetric mean squared Chamfer in units of
    distance_scale meters; semantic cost is positive-class sigmoid NLL,
    matching the official independent class-logit convention. Endpoints are
    equally averaged within a track. SciPy Hungarian sees detached costs,
    but each mode's selected whole-track costs retain their autograd graph.
    Work is allocated one track at a time, never as [K,M,F,Q,P,S].
    """
    if mode_points.ndim != 5 or mode_points.shape[-1] != 3:
        raise ValueError('mode_points must have shape [K,F,Q,P,3]')
    if mode_scores.ndim != 5 or mode_scores.shape[:-1] != mode_points.shape[:-1]:
        raise ValueError('mode_scores must match [K,F,Q,P,C]')
    if max_pred_points < 1 or max_tracks < 1 or distance_scale <= 0:
        raise ValueError('Sampling limits and distance_scale must be positive')
    if geometry_weight < 0 or semantic_weight < 0:
        raise ValueError('Matching weights must be nonnegative')
    modes, horizons, queries, pred_count, _ = mode_points.shape
    if not modes or not queries or not pred_count:
        raise ValueError('Modes, queries, and predicted point sets must be nonempty')
    device = mode_points.device
    points = torch.as_tensor(targets['points'], device=device, dtype=torch.float32)
    valid = torch.as_tensor(targets['point_valid'], device=device, dtype=torch.bool)
    observed = torch.as_tensor(targets['observed'], device=device, dtype=torch.bool)
    labels = torch.as_tensor(targets['labels'], device=device, dtype=torch.long)
    if points.ndim != 4 or points.shape[1] != horizons or points.shape[-1] != 3:
        raise ValueError('Target points must have shape [M,F,S,3]')
    if valid.shape != points.shape[:-1] or observed.shape != points.shape[:2] or labels.shape != points.shape[:1]:
        raise ValueError('Target masks and labels do not match points')
    if len(points) > max_tracks or points.shape[2] > 16:
        raise ValueError('Endpoint cache must apply the declared max_tracks/S<=16 cap')
    if ((labels < 0) | (labels >= mode_scores.shape[-1])).any():
        raise ValueError('An endpoint label is not an occupied semantic class')
    # NaN padding is legal only where the endpoint/point is explicitly unknown.
    valid = valid & observed[..., None]
    if not torch.isfinite(points[valid]).all():
        raise ValueError('A valid observed endpoint contains nonfinite coordinates')
    seen = valid.any(-1) & observed
    usable = (seen.sum(-1) >= 2) & seen[:, 1:].any(-1)
    track_indices = torch.nonzero(usable, as_tuple=False).flatten()
    if len(track_indices) > queries:
        raise ValueError('One-to-one matching requires at least one query per track')
    sample = torch.linspace(0, pred_count - 1, min(pred_count, max_pred_points), device=device).long()
    predicted = mode_points.index_select(3, sample).float()
    scores = mode_scores.index_select(3, sample).float()
    # Retain both graphs even for a scene without any eligible real track.
    zero = (predicted.sum(dim=(1, 2, 3, 4)) + scores.sum(dim=(1, 2, 3, 4))) * 0.
    if not len(track_indices):
        if return_details:
            return zero, {'track_indices': track_indices, 'assignments': [], 'cost_matrices': []}
        return zero

    # Indices, visibility and labels carry no gradients. Transfer them together
    # once, rather than synchronizing CUDA separately for every mode/track.
    metadata = torch.cat((track_indices[:, None],
                          labels.index_select(0, track_indices)[:, None],
                          seen.index_select(0, track_indices).long()), dim=1).detach().cpu().tolist()
    track_records = [(row[0], row[1], [frame for frame, present in enumerate(row[2:]) if present])
                     for row in metadata]
    from scipy.optimize import linear_sum_assignment
    energies, assignments, matrices = [], [], []
    for mode in range(modes):
        track_costs = []
        for track_index, label, frames in track_records:
            frame_costs = []
            for frame in frames:
                truth = points[track_index, frame, valid[track_index, frame]]
                prediction = predicted[mode, frame]
                # cdist is float32 here even when the outer training uses AMP.
                with torch.cuda.amp.autocast(enabled=False):
                    squared = torch.cdist(prediction, truth[None].expand(queries, -1, -1)).square()
                    geometry = .5 * (squared.min(-1).values.mean(-1) + squared.min(-2).values.mean(-1))
                    semantic = F.softplus(-scores[mode, frame, :, :, label]).mean(-1)
                    frame_costs.append(geometry_weight * geometry / distance_scale ** 2 + semantic_weight * semantic)
            track_costs.append(torch.stack(frame_costs).mean(0))
        costs = torch.stack(track_costs)
        if not torch.isfinite(costs).all():
            raise FloatingPointError('Nonfinite whole-track matching costs')
        row, column = linear_sum_assignment(costs.detach().cpu().numpy())
        row = torch.as_tensor(row, device=device, dtype=torch.long)
        column = torch.as_tensor(column, device=device, dtype=torch.long)
        energies.append(costs[row, column].mean() + zero[mode])
        if return_details:
            assignments.append((track_indices[row], column))
            matrices.append(costs.detach())
    energy = torch.stack(energies)
    if return_details:
        return energy, {'track_indices': track_indices, 'assignments': assignments, 'cost_matrices': matrices}
    return energy
