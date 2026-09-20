"""One reference-time motion posterior, reused by every layer and horizon.

The legacy cache supplies a compensated radial *proxy*, not raw Doppler or
sensor quality fields. Information increments are constructed from its LOS.
This is a bounded, robust Gaussian approximation, not calibrated occupancy.
"""
import torch
from torch import nn


def inverse_2x2(matrix):
    """Analytic SPD inverse avoids repeated cuSOLVER launches for tiny matrices."""
    a, b, c, d = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 0], matrix[..., 1, 1]
    determinant = a * d - b * c
    inverse = torch.stack([d, -b, -c, a], -1).reshape(*matrix.shape)
    return inverse / determinant.clamp_min(1e-12)[..., None, None], determinant


def information_update(mean, covariance, directions, radial, variance, weight):
    """Rank-one observation information; no evidence returns the exact prior."""
    with torch.cuda.amp.autocast(enabled=False):
        mean, covariance = mean.float(), covariance.float()
        n, d = directions.float(), radial.float()
        precision = weight.float() / variance.float().clamp_min(.25)
        information = (precision[..., None, None] * n[..., :, None] * n[..., None, :]).sum(-3)
        natural = (precision[..., None] * n * d[..., None]).sum(-2)
        prior_precision = inverse_2x2(covariance)[0]
        posterior = inverse_2x2(prior_precision + information)[0]
        posterior = (posterior + posterior.transpose(-1, -2)) * .5
        updated = (posterior @ ((prior_precision @ mean[..., None]).squeeze(-1) + natural)[..., None]).squeeze(-1)
        present = weight.sum(-1) > 0
        return (torch.where(present[..., None], updated, mean),
                torch.where(present[..., None, None], posterior, covariance), information)


def propagated_position(centers, mean, covariance, seconds, position_std=2., acceleration_std=.5):
    if seconds < 0:
        raise ValueError('Negative forecast horizon')
    with torch.cuda.amp.autocast(enabled=False):
        eye = torch.eye(2, device=centers.device)
        position = centers.float()[..., :2] + seconds * mean.float()
        spread = seconds ** 2 * covariance.float() + (
            position_std ** 2 + .25 * seconds ** 4 * acceleration_std ** 2) * eye
        return position, spread


def holdout_groups(points):
    """Split spatial/time groups independently of observed velocity values."""
    cell = torch.floor(points[:, :2] / 2.).long()
    time = torch.floor(points[:, 6] / .1).long()
    return ((cell[:, 0] * 73856093 + cell[:, 1] * 19349663 + time * 83492791) % 5) == 0


def network(net, values):
    return net(values.to(dtype=next(net.parameters()).dtype)).float()


class SharedRadarBelief(nn.Module):
    def __init__(self, embed_dims=256, neighbors=16, radius=6., chunk_size=128,
                 heldout_weight=.05):
        super().__init__()
        self.neighbors, self.radius, self.chunk_size = neighbors, radius, chunk_size
        self.heldout_weight = heldout_weight
        self.geometry_velocity = True
        self.feature_velocity = True
        self.prior = nn.Sequential(nn.LayerNorm(embed_dims), nn.Linear(embed_dims, 128),
                                   nn.SiLU(), nn.Linear(128, 5))
        nn.init.normal_(self.prior[-1].weight, std=.001)
        nn.init.zeros_(self.prior[-1].bias)
        self.noise = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.noise[-1].weight)
        nn.init.zeros_(self.noise[-1].bias)
        self.point_encoder = nn.Sequential(nn.Linear(8, embed_dims), nn.LayerNorm(embed_dims),
                                            nn.ReLU(), nn.Linear(embed_dims, embed_dims))
        self.state_encoder = nn.Sequential(nn.Linear(6, embed_dims), nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims))

    def prior_state(self, features):
        raw = network(self.prior, features)
        mean = 35. * (raw[..., :2] / 35.).tanh()
        std = .5 + 9.5 * raw[..., 2:4].sigmoid()
        rho = .8 * raw[..., 4].tanh()
        cross = rho * std[..., 0] * std[..., 1]
        covariance = torch.stack([std[..., 0] ** 2, cross, cross, std[..., 1] ** 2], -1)
        return mean, covariance.reshape(*mean.shape[:-1], 2, 2)

    def assimilate(self, centers, prior_mean, prior_cov, points, variance, features, allowed):
        means, covs, pooled, supported, strengths = [], [], [], [], []
        for start in range(0, len(centers), self.chunk_size):
            end = start + self.chunk_size
            x, mu, cov = centers[start:end].float(), prior_mean[start:end], prior_cov[start:end]
            # Associate at each return's historical timestamp, not at future time.
            with torch.cuda.amp.autocast(enabled=False):
                historical = x[:, None, :2] - mu[:, None] * points[None, :, 6:7]
                distance = (historical - points[None, :, :2]).square().sum(-1)
                distance = distance.masked_fill(~allowed[None], float('inf'))
                distance, index = distance.topk(min(self.neighbors, len(points)), largest=False)
                neighbor = points[index]
                residual = neighbor[..., 7] - (neighbor[..., 8:10] * mu[:, None]).sum(-1)
                robust = (1. + (residual.detach() / 5.).square()).reciprocal()
                raw = torch.exp(-distance / (self.radius ** 2 / 2)) * robust
                raw = raw * (distance <= self.radius ** 2) * allowed[index]
                mass = raw.sum(-1, keepdim=True)
                association = raw / mass.clamp_min(1e-8)
                # One effective observation budget. Correlated points cannot
                # add unbounded information simply by increasing return count.
                weight = association * mass.clamp(max=1.)
                mean, covariance, _ = information_update(
                    mu, cov, neighbor[..., 8:10], neighbor[..., 7], variance[index], weight)
                strength = (1. - covariance.diagonal(dim1=-2, dim2=-1).sum(-1) /
                            cov.diagonal(dim1=-2, dim2=-1).sum(-1)).clamp(0., 1.)
                value = (features[index] * association[..., None]).sum(-2)
            means.append(mean); covs.append(covariance); pooled.append(value)
            supported.append(mass[:, 0] > .05); strengths.append(strength)
        return dict(mean=torch.cat(means), covariance=torch.cat(covs),
                    features=torch.cat(pooled), support=torch.cat(supported),
                    strength=torch.cat(strengths), centers=centers.float())

    def heldout_loss(self, centers, prior_mean, posterior, points, variance, heldout):
        targets = points[heldout]
        if not len(targets) or not posterior['support'].any():
            return posterior['mean'].sum() * 0., 0
        with torch.cuda.amp.autocast(enabled=False):
            # Association uses the camera prior and target locations/age only;
            # target radial values never enter the assimilated posterior.
            historical = centers[None, :, :2].float() - prior_mean[None] * targets[:, None, 6:7]
            distance = (historical - targets[:, None, :2]).square().sum(-1)
            distance = distance.masked_fill(~posterior['support'][None], float('inf'))
            distance, index = distance.topk(min(4, len(centers)), largest=False)
            weight = torch.exp(-distance / (self.radius ** 2 / 2)) * (distance <= self.radius ** 2)
            present = weight.sum(-1) > .05
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            n = targets[:, None, 8:10]
            component_mean = (n * posterior['mean'][index]).sum(-1)
            component_var = (n[..., None, :] @ posterior['covariance'][index] @ n[..., :, None]).squeeze(-1).squeeze(-1)
            prediction = (weight * component_mean).sum(-1)
            predictive_variance = (weight * (component_var + (component_mean - prediction[:, None]).square())).sum(-1)
            predictive_variance = predictive_variance + variance[heldout]
            nll = .5 * ((targets[:, 7] - prediction).square() / predictive_variance.clamp_min(.25) +
                        predictive_variance.clamp_min(.25).log())
            return (nll * present).sum() / present.sum().clamp_min(1), int(present.sum())

    def forward(self, query, centers, radar, valid):
        prior_mean, prior_cov = self.prior_state(query)
        states, losses, counts = [], [], []
        for b in range(len(query)):
            points = radar[b][valid[b]].float()
            reliable = (torch.isfinite(points).all(-1) & (points[:, 6] >= 0) &
                        (points[:, 6] <= .5) & (points[:, 7].abs() <= 35.) &
                        (points[:, 8:10].square().sum(-1) > 1e-6))
            points = torch.unique(points[reliable], dim=0)  # exact duplicates assimilated once
            if not len(points):
                state = dict(mean=prior_mean[b], covariance=prior_cov[b], centers=centers[b].float(),
                             features=query[b].float() * 0., support=valid.new_zeros(query.shape[1]),
                             strength=query[b, :, 0].float() * 0.)
                states.append(state); losses.append(prior_mean[b].sum() * 0.); counts.append(0)
                continue
            noise_input = torch.stack([points[:, :2].norm(dim=-1) / 40., points[:, 5] / 20.,
                                       points[:, 6] * 2., points[:, 8:10].norm(dim=-1)], -1)
            variance = (.5 + 3.5 * network(self.noise, noise_input).sigmoid().squeeze(-1)).square()
            point_input = points[:, [0, 1, 2, 5, 6, 7, 8, 9]].clone()
            if not self.feature_velocity:
                point_input[:, 5] = 0
            point_input /= points.new_tensor([40., 40., 5., 20., 1., 20., 1., 1.])
            features = network(self.point_encoder, point_input)
            allowed = torch.ones(len(points), dtype=torch.bool, device=points.device)
            state = self.assimilate(centers[b], prior_mean[b], prior_cov[b], points, variance, features, allowed)
            if self.training and self.heldout_weight:
                heldout = holdout_groups(points)
                partial = self.assimilate(centers[b], prior_mean[b], prior_cov[b], points,
                                          variance, features * 0., ~heldout)
                loss, count = self.heldout_loss(centers[b], prior_mean[b], partial, points, variance, heldout)
            else:
                loss, count = state['mean'].sum() * 0., 0
            covariance = state['covariance']
            summary = torch.cat([state['mean'] / 20., covariance[:, 0, :1] / 100.,
                                 covariance[:, 0, 1:] / 100., covariance[:, 1, 1:] / 100.,
                                 state['strength'][:, None]], -1)
            if not self.feature_velocity:
                summary = summary * 0.
            state['features'] = state['features'] + network(self.state_encoder, summary)
            states.append(state); losses.append(loss); counts.append(count)
        state = {key: torch.stack([s[key] for s in states]) for key in states[0]}
        state['geometry_velocity'] = self.geometry_velocity
        state['heldout_count'] = sum(counts)
        # Equal scene weighting; absent held-out support contributes zero.
        auxiliary = torch.stack(losses).mean() * self.heldout_weight
        return state, auxiliary


class BeliefReadout(nn.Module):
    """Spatial lookup of a shared posterior; never assume future query identity."""
    mode = 'belief'

    def __init__(self, embed_dims=256, neighbors=8, chunk_size=128):
        super().__init__()
        self.neighbors, self.chunk_size = neighbors, chunk_size
        self.gate = nn.Linear(2 * embed_dims, embed_dims)
        self.output = nn.Linear(embed_dims, embed_dims, bias=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, centers, state, seconds):
        h = seconds if state['geometry_velocity'] else 0.
        position, covariance = propagated_position(state['centers'], state['mean'], state['covariance'], h)
        outputs = []
        for start in range(0, query.shape[1], self.chunk_size):
            end = start + self.chunk_size
            with torch.cuda.amp.autocast(enabled=False):
                delta = centers[:, start:end, None, :2].float() - position[:, None]
                distance = delta.square().sum(-1).masked_fill(~state['support'][:, None], float('inf'))
                _, index = distance.topk(min(self.neighbors, position.shape[1]), largest=False)
                batch = torch.arange(len(query), device=query.device)[:, None, None]
                diff = centers[:, start:end, None, :2].float() - position[batch, index]
                cov = covariance[batch, index]
                mahalanobis = (diff[..., None, :] @ inverse_2x2(cov)[0] @ diff[..., :, None]).squeeze(-1).squeeze(-1)
                # Density and information reduction suppress diffuse evidence;
                # increasing uncertainty does not simply enlarge occupied area.
                weight = torch.exp(-.5 * mahalanobis) * (mahalanobis <= 9.)
                weight = weight * (4. / inverse_2x2(cov)[1].clamp_min(1e-6).sqrt())
                weight = weight * state['strength'][batch, index] * state['support'][batch, index]
                mass = weight.sum(-1, keepdim=True)
                pooled = (state['features'][batch, index] * weight[..., None]).sum(-2) / mass.clamp_min(1e-8)
            q = query[:, start:end]
            update = self.output(pooled.to(self.output.weight.dtype))
            gate_input = torch.cat([q, pooled.to(q.dtype)], -1).to(self.gate.weight.dtype)
            update = update * self.gate(gate_input).sigmoid()
            outputs.append(q + (update.float() * mass.clamp(max=1.)).to(q.dtype))
        return torch.cat(outputs, dim=1)
