# Third-party dependencies

This project is based on **3D Diffusion Policy (DP3)** by Yanjie Ze and collaborators:
https://github.com/YanjieZe/3D-Diffusion-Policy

The MOP policy imports DP3's `PointNetEncoderXYZ`, `ConditionalUnet1D`,
`LinearNormalizer`, and `BasePolicy` directly from `diffusion_policy_3d`.
DP3 is distributed under the MIT License. The upstream license is retained in
`licenses/DP3-MIT.txt`. The compatibility reference is upstream commit
`47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`.

The RGB backbone uses TorchVision's ResNet-18. Diffusion schedules are implemented
with Hugging Face Diffusers. These packages retain their respective licenses.
Pretrained image weights are optional and are not distributed here.
