# MOP: Multimodal Object-Aware Policy

### Robotic Manipulation via Geometry-Guided Fusion and Trajectory Prediction

**IEEE Robotics and Automation Letters · 2026**

[**Project Page**](https://reshmouqi01.github.io/MOPpage/) · [**Paper**](https://doi.org/10.1109/LRA.2026.3710341) · [**DP3 Backbone**](https://github.com/YanjieZe/3D-Diffusion-Policy)

**MOP is built on [3D Diffusion Policy (DP3)](https://github.com/YanjieZe/3D-Diffusion-Policy).** It extends point-cloud-based action diffusion with geometry-guided RGB fusion and object-trajectory conditioning. This repository contains the core policy modules, an aligned-data training example, and the project-page resources.

![MOP framework](static/images/overall.png)

## Method

MOP combines three components in a jointly trained policy:

1. **Geometry-guided multimodal perception (G2MP).** ResNet-18 features are reduced to spatial-softmax keypoints. A DP3 PointNet encodes XYZ geometry, which supplies the cross-attention query; image features supply keys and values. A learnable residual gate controls the visual contribution, while modality dropout preserves a geometry-based path. The fused representation is concatenated with the encoded robot state.
2. **Object position prediction (OPP).** A Transformer CVAE predicts future object XYZ trajectories. During training, its posterior encoder receives observations and labeled future positions. At inference, the latent variable is sampled from the standard-normal prior. Learned object queries attend to the observation/latent memory to decode the trajectory.
3. **Trajectory-conditioned action diffusion.** A second residual gate injects predicted object motion into the observation representation. DP3's conditional 1D U-Net receives this representation through FiLM and denoises an action sequence for receding-horizon execution.

```text
RGB → ResNet-18 → spatial keypoints ───────────────┐
                                                ↓ keys / values
XYZ → DP3 PointNet → geometry query → gated cross-attention
                                                ↓
Robot state ───────────────────────────→ observation features
                                                ↓
                              Transformer CVAE → object trajectory
                                                ↓ residual gate
                               DP3 FiLM U-Net → action sequence
```

The joint objective combines diffusion noise-prediction MSE, object-position L1 reconstruction and latent KL divergence, and squared regularization of the two residual gates:

```text
L = L_diffusion + λ_opp · (L_position + β_KL · L_KL) + λ_reg · L_gates
```

Object trajectories supervise OPP during training; action inference takes only the observation window. Simulation labels can come from simulator object states. For RGB-D recordings, the geometry utilities lift externally tracked image points into the point-cloud coordinate frame and aggregate visible tracks into an object-position target.

## Core code

```text
mop/
  vision.py          # spatial-softmax ResNet + DP3 PointNet + geometry-query fusion
  trajectory.py      # posterior/prior CVAE, learned queries, trajectory residual gate
  policy.py          # DP3-compatible predict_action / compute_loss interfaces
  losses.py          # diffusion, position, KL, and gate objectives
  geometry.py        # RGB-D track lifting and visible-track aggregation
configs/mop.yaml     # policy and optimizer configuration
examples/train_core.py
tests/test_core.py
```

The DP3 implementation remains an external dependency. This policy directly reuses its **`PointNetEncoderXYZ`**, **`ConditionalUnet1D`**, **`LinearNormalizer`**, and **`BasePolicy`**. Attribution and the upstream license are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/DP3-MIT.txt](licenses/DP3-MIT.txt).

## Installation

Use Python 3.10 or newer with a PyTorch/TorchVision combination appropriate for your environment. From this repository's root:

```bash
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git
git -C 3D-Diffusion-Policy checkout 47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97
python -m pip install -e ./3D-Diffusion-Policy/3D-Diffusion-Policy
python -m pip install -e .
```

The configuration exposes the image-backbone initialization through `pretrained_rgb`. Enabling it uses TorchVision's ImageNet ResNet-18 weights.

## Data interface

Each training sample is a temporally aligned window:

- `obs.rgb`: `(B, 2, 3, H, W)`, RGB `uint8` in `[0,255]` or floating point in `[0,1]`.
- `obs.point_cloud`: `(B, 2, N, 3)`, finite XYZ coordinates in a consistent scene frame.
- `obs.agent_pos`: `(B, 2, state_dim)`, robot proprioception.
- `action`: `(B, 16, action_dim)`, the DP3-aligned action sequence.
- `object_position`: `(B, 16, 3)`, future object positions expressed in the same frame and units as the point cloud.
- Optional `object_valid` and `action_valid`: boolean `(B,16)` masks for unavailable trajectory labels or padded actions.

Fit the `LinearNormalizer` on training data only. Point clouds and object positions share the same XYZ normalization; robot states and actions have separate statistics. The execution chunk begins at action index `n_obs_steps - 1`, following DP3's sequence convention.

The training example reads an NPZ archive with keys `rgb`, `point_cloud`, `agent_pos`, `action`, and `object_position`, plus optional validity masks. Its leading dimension is the number of training windows:

```bash
python -m examples.train_core --data data/training_windows.npz --config configs/mop.yaml --epochs 1000
```

Dataset adapters provide the synchronized image, cloud, robot, action, and object-position windows. Task-specific `state_dim` and `action_dim` are inferred from the archive.

## Configuration

[configs/mop.yaml](configs/mop.yaml) includes the main settings from the paper's Table I:

- Observation / action prediction / execution horizons: **2 / 16 / 8**.
- Object-position prediction horizon: **16**; geometric feature dimension: **128**.
- Diffusion hidden dimensions: **512, 1024, 2048**.
- Training / inference diffusion steps: **100 / 10**.
- Batch size: **128**; learning rate: **1 × 10⁻⁴**.
- Auxiliary and gate-regularization weights: **λ_opp = 1**, **λ_reg = 0.01**.

The configuration also exposes the state embedding, CVAE dimensions, modality-dropout probability, gate initialization, and KL weight.

## Inference

```python
import torch
from mop.policy import MOPPolicy

checkpoint = torch.load("checkpoints/mop.pt", map_location="cpu", weights_only=True)
policy = MOPPolicy(
    state_dim=checkpoint["state_dim"],
    action_dim=checkpoint["action_dim"],
    **checkpoint["policy_config"],
)
policy.load_state_dict(checkpoint["model"])
policy.eval()

# observation_window contains rgb, point_cloud, and agent_pos tensors.
prediction = policy.predict_action(observation_window)
actions = prediction["action"]                # (B, 8, action_dim)
full_sequence = prediction["action_pred"]     # (B, 16, action_dim)
object_xyz = prediction["object_position"]    # (B, 16, 3)
```

For GPU inference, move the policy and observation tensors to the same device. A `torch.Generator` can be passed to `predict_action` to control prior and diffusion sampling.

## Tests

```bash
python -m pip install pytest
python -m pytest tests -q
```

Tests cover spatial keypoints, modality dropout, gradient flow through the joint objective, padded targets, prior-based inference, checkpoint normalization, and RGB-D coordinate lifting.

## Citation

Please cite MOP and the DP3 backbone when using this work:

```bibtex
@article{zhang2026mop,
  title={MOP: A Multimodal Object-Aware Policy for Robotic Manipulation
         via Geometry-Guided Fusion and Trajectory Prediction},
  author={Zhang, Keyu and Yang, Yicheng and Wang, Lifeng and Li, Ruijiao
          and Ding, Jie and Li, Cong and Shang, Huiliang and Xu, Yin},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={9},
  pages={10114--10121},
  year={2026},
  doi={10.1109/LRA.2026.3710341}
}
```

DP3 citation and project resources: [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy).

## License

Core MOP code is released under the [MIT License](LICENSE). External dependencies retain their respective licenses.
