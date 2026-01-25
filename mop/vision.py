"""ResNet spatial keypoints and geometry-query multimodal perception."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Spatial-softmax temperature must be positive.")
        self.temperature = temperature

    def forward(self, features):
        """(B,C,H,W) -> (B,C,2), coordinates ordered as (x,y) in [-1,1]."""
        _, _, height, width = features.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1, 1, width, device=features.device, dtype=features.dtype),
            indexing="ij")
        coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1)
        weights = (features.flatten(2) / self.temperature).softmax(dim=-1)
        return weights @ coordinates


class ImageKeypointEncoder(nn.Module):
    def __init__(self, pretrained=False, temperature=1.0):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = SpatialSoftmax(temperature)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB input must be (B,3,H,W).")
        is_uint8 = rgb.dtype == torch.uint8
        rgb = rgb.to(self.mean.dtype)
        if is_uint8:
            rgb = rgb / 255
        return self.pool(self.backbone((rgb - self.mean) / self.std))


class G2MPEncoder(nn.Module):
    """Geometry is the residual path and attention query; image keypoints are keys/values."""
    def __init__(self, state_dim, feature_dim=128, state_feature_dim=64,
                 heads=4, modality_dropout=.1, gate_init=.1, pretrained_rgb=False):
        super().__init__()
        if not 0 <= modality_dropout <= 1:
            raise ValueError("Modality dropout must be in [0,1].")
        self.point_encoder = PointNetEncoderXYZ(
            in_channels=3, out_channels=feature_dim, use_layernorm=True, final_norm="layernorm")
        self.image_encoder = ImageKeypointEncoder(pretrained=pretrained_rgb)
        self.visual_projection = nn.Linear(2, feature_dim, bias=False)
        self.attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim), nn.ReLU(),
            nn.Linear(state_feature_dim, state_feature_dim))
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))
        self.modality_dropout = modality_dropout
        self.output_dim = feature_dim + state_feature_dim

    def forward(self, point_cloud, rgb, state):
        geometric = self.point_encoder(point_cloud)
        image_tokens = self.visual_projection(self.image_encoder(rgb))
        visual, _ = self.attention(geometric[:, None], image_tokens, image_tokens, need_weights=False)
        visual = visual[:, 0]
        if self.training and self.modality_dropout:
            keep = torch.rand((len(visual), 1), device=visual.device) >= self.modality_dropout
            visual = visual * keep.to(visual.dtype)
        fused = geometric + self.alpha * visual
        return torch.cat((fused, self.state_encoder(state)), dim=-1)
