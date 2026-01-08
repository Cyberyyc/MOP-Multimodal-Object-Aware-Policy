"""Transformer CVAE for future object positions and residual policy conditioning."""
import torch
from torch import nn


class ObjectQueryBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width * 4, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, memory_position):
        attended, _ = self.self_attention(query, query, query, need_weights=False)
        query = self.norms[0](query + self.dropout(attended))
        attended, _ = self.cross_attention(query, memory + memory_position, memory, need_weights=False)
        query = self.norms[1](query + self.dropout(attended))
        return self.norms[2](query + self.dropout(self.ffn(query)))


class ObjectPositionPredictor(nn.Module):
    def __init__(self, observation_dim, observation_steps=2, horizon=16,
                 width=128, latent_dim=32, heads=4, layers=2, dropout=.1):
        super().__init__()
        self.observation_steps, self.horizon, self.latent_dim = observation_steps, horizon, latent_dim
        self.observation_projection = nn.Linear(observation_dim, width)
        self.position_projection = nn.Linear(3, width)
        self.latent_projection = nn.Linear(latent_dim, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.encoder_position = nn.Parameter(torch.randn(1, 1 + observation_steps + horizon, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.posterior_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.posterior_parameters = nn.Linear(width, 2 * latent_dim)
        self.object_queries = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.memory_position = nn.Parameter(torch.randn(1, observation_steps + 1, width) * .02)
        self.decoder = nn.ModuleList([ObjectQueryBlock(width, heads, dropout) for _ in range(layers)])
        self.output = nn.Linear(width, 3)

    def forward(self, observations, target=None, valid=None, generator=None, sample=True):
        batch, steps, _ = observations.shape
        if steps != self.observation_steps:
            raise ValueError("Observation history does not match OPP configuration.")
        observed = self.observation_projection(observations)
        mean = logvar = None
        if target is not None:
            if target.shape != (batch, self.horizon, 3):
                raise ValueError("Object targets must be (B,object_horizon,3).")
            valid = torch.ones(target.shape[:2], device=target.device, dtype=torch.bool) if valid is None else valid.bool()
            if valid.shape != target.shape[:2] or not valid.any(dim=1).all():
                raise ValueError("Every trajectory requires at least one valid target.")
            clean_target = target.masked_fill(~valid[..., None], 0)
            if not torch.isfinite(clean_target).all():
                raise ValueError("Visible object targets must be finite.")
            sequence = torch.cat((self.cls.expand(batch, -1, -1), observed,
                                  self.position_projection(clean_target)), dim=1)
            padding = torch.cat((torch.zeros((batch, steps + 1), device=target.device, dtype=torch.bool), ~valid), dim=1)
            posterior = self.posterior_encoder(sequence + self.encoder_position, src_key_padding_mask=padding)[:, 0]
            mean, logvar = self.posterior_parameters(posterior).chunk(2, dim=-1)
            logvar = logvar.clamp(-20, 10)
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            latent = mean + torch.exp(.5 * logvar) * noise if sample else mean
        else:
            shape = (batch, self.latent_dim)
            latent = torch.randn(shape, device=observations.device, dtype=observations.dtype,
                                 generator=generator) if sample else observations.new_zeros(shape)
        memory = torch.cat((observed, self.latent_projection(latent)[:, None]), dim=1)
        query = self.object_queries.expand(batch, -1, -1)
        for block in self.decoder:
            query = block(query, memory, self.memory_position)
        return self.output(query), mean, logvar


class TrajectoryConditioner(nn.Module):
    def __init__(self, observation_dim, observation_steps=2, horizon=16, gate_init=.1):
        super().__init__()
        condition_dim = observation_dim * observation_steps
        self.beta = nn.Parameter(torch.tensor(float(gate_init)))
        self.residual = nn.Sequential(nn.Linear(condition_dim + horizon * 3, condition_dim),
                                      nn.ReLU(), nn.Linear(condition_dim, condition_dim))

    def forward(self, observations, trajectory):
        observation = observations.flatten(1)
        return observation + self.beta * self.residual(torch.cat((trajectory.flatten(1), observation), dim=-1))
