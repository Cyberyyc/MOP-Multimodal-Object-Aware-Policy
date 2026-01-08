"""DP3-based MOP policy: G2MP -> OPP -> FiLM-conditioned action diffusion."""
import torch
from diffusers import DDPMScheduler
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from .vision import G2MPEncoder
from .trajectory import ObjectPositionPredictor, TrajectoryConditioner
from .losses import position_cvae_loss, gate_sparsity, diffusion_mse


class MOPPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, horizon=16, n_obs_steps=2, n_action_steps=8,
                 object_horizon=16, feature_dim=128, state_feature_dim=64,
                 transformer_width=128, latent_dim=32, transformer_heads=4,
                 transformer_layers=2, modality_dropout=.1, gate_init=.1,
                 pretrained_rgb=False, down_dims=(512, 1024, 2048),
                 diffusion_steps=100, inference_steps=10, lambda_opp=1., lambda_reg=.01,
                 kl_weight=.01, noise_scheduler=None):
        super().__init__()
        if min(state_dim, action_dim, n_obs_steps, n_action_steps, object_horizon) <= 0:
            raise ValueError("Dimensions and horizons must be positive.")
        if n_obs_steps - 1 + n_action_steps > horizon or horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("Action horizon must fit the execution slice and U-Net downsampling.")
        self.action_dim, self.state_dim = action_dim, state_dim
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.object_horizon = object_horizon
        self.encoder = G2MPEncoder(state_dim, feature_dim, state_feature_dim,
                                   transformer_heads, modality_dropout, gate_init, pretrained_rgb)
        self.opp = ObjectPositionPredictor(self.encoder.output_dim, n_obs_steps, object_horizon,
                                           transformer_width, latent_dim, transformer_heads, transformer_layers)
        self.conditioner = TrajectoryConditioner(self.encoder.output_dim, n_obs_steps, object_horizon, gate_init)
        self.model = ConditionalUnet1D(input_dim=action_dim, local_cond_dim=None,
                                       global_cond_dim=self.encoder.output_dim * n_obs_steps,
                                       down_dims=list(down_dims), condition_type="film", kernel_size=5,
                                       use_down_condition=True, use_mid_condition=True, use_up_condition=True)
        self.scheduler = noise_scheduler if noise_scheduler is not None else DDPMScheduler(
            num_train_timesteps=diffusion_steps, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("MOP uses epsilon prediction for the diffusion objective.")
        if not 0 < inference_steps <= self.scheduler.config.num_train_timesteps:
            raise ValueError("Invalid diffusion inference step count.")
        self.inference_steps = inference_steps
        self.lambda_opp, self.lambda_reg, self.kl_weight = lambda_opp, lambda_reg, kl_weight
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(self.device)

    def encode_observations(self, obs):
        pc, rgb, state = (obs[key] for key in ("point_cloud", "rgb", "agent_pos"))
        if pc.ndim != 4 or pc.shape[-1] != 3 or pc.shape[2] < 1:
            raise ValueError("Point clouds must be nonempty (B,T,N,3) XYZ arrays.")
        batch, steps = pc.shape[:2]
        if steps != self.n_obs_steps or state.shape != (batch, steps, self.state_dim):
            raise ValueError("State/history shape does not match policy configuration.")
        if rgb.ndim != 5 or rgb.shape[:3] != (batch, steps, 3):
            raise ValueError("RGB must be (B,T,3,H,W).")
        pc = self.normalizer['point_cloud'].normalize(pc)
        state = self.normalizer['agent_pos'].normalize(state)
        return self.encoder(pc.flatten(0, 1), rgb.flatten(0, 1), state.flatten(0, 1)).reshape(batch, steps, -1)

    def compute_loss(self, batch):
        observations = self.encode_observations(batch['obs'])
        action = self.normalizer['action'].normalize(batch['action'])
        if action.shape != (len(observations), self.horizon, self.action_dim):
            raise ValueError("Actions must match the full prediction horizon.")
        target = self.normalizer['point_cloud'].normalize(batch['object_position'])
        trajectory, mean, logvar = self.opp(observations, target, batch.get('object_valid'))
        condition = self.conditioner(observations, trajectory)
        action_valid = batch.get('action_valid')
        if action_valid is not None:
            if action_valid.shape != action.shape[:2]:
                raise ValueError("Action-valid mask must be (B,action_horizon).")
            action = action.masked_fill(~action_valid.bool()[..., None], 0)
        noise = torch.randn_like(action)
        timesteps = torch.randint(self.scheduler.config.num_train_timesteps, (len(action),), device=action.device)
        noisy_action = self.scheduler.add_noise(action, noise, timesteps)
        prediction = self.model(noisy_action, timesteps, global_cond=condition)
        diffusion = diffusion_mse(prediction, noise, action_valid)
        opp, reconstruction, kl = position_cvae_loss(trajectory, target, mean, logvar,
                                                    batch.get('object_valid'), self.kl_weight)
        sparse = gate_sparsity(self.encoder.alpha, self.conditioner.beta)
        loss = diffusion + self.lambda_opp * opp + self.lambda_reg * sparse
        metrics = {"diffusion": diffusion, "opp": opp, "position_l1": reconstruction,
                   "kl": kl, "sparsity": sparse, "alpha": self.encoder.alpha, "beta": self.conditioner.beta}
        return loss, {key: float(value.detach()) for key, value in metrics.items()}

    @torch.no_grad()
    def predict_action(self, obs_dict, generator=None):
        if self.training:
            raise RuntimeError("Call policy.eval() before action inference.")
        observations = self.encode_observations(obs_dict)
        trajectory, _, _ = self.opp(observations, generator=generator)
        condition = self.conditioner(observations, trajectory)
        action = torch.randn((len(observations), self.horizon, self.action_dim),
                             device=condition.device, dtype=condition.dtype, generator=generator)
        self.scheduler.set_timesteps(self.inference_steps, device=condition.device)
        for step in self.scheduler.timesteps:
            epsilon = self.model(action, step, global_cond=condition)
            action = self.scheduler.step(epsilon, step, action, generator=generator).prev_sample
        action = self.normalizer['action'].unnormalize(action)
        start = self.n_obs_steps - 1
        return {'action': action[:, start:start + self.n_action_steps], 'action_pred': action,
                'object_position': self.normalizer['point_cloud'].unnormalize(trajectory)}
