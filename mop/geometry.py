"""Lift externally tracked image points into the RGB-D scene coordinate frame."""
import numpy as np


def lift_tracks_to_3d(pixels, depth, intrinsics, visible=None, camera_to_world=None, depth_scale=1.):
    """pixels (T,K,2), depth (T,H,W); returns XYZ (T,K,3) and a valid-depth mask."""
    pixels, depth = np.asarray(pixels), np.asarray(depth)
    intrinsics = np.asarray(intrinsics, dtype=float)
    if pixels.ndim != 3 or pixels.shape[-1] != 2 or depth.ndim != 3 or len(depth) != len(pixels):
        raise ValueError("Expected time-aligned tracks and depth images.")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or depth_scale <= 0:
        raise ValueError("Provide finite intrinsics and a positive depth conversion scale.")
    finite = np.isfinite(pixels).all(-1)
    safe = np.where(finite[..., None], pixels, 0)
    x, y = np.rint(safe).astype(np.int64).transpose(2, 0, 1)
    valid = finite & (x >= 0) & (x < depth.shape[2]) & (y >= 0) & (y < depth.shape[1])
    z = depth[np.arange(len(depth))[:, None], y.clip(0, depth.shape[1]-1), x.clip(0, depth.shape[2]-1)] * depth_scale
    valid &= np.isfinite(z) & (z > 0)
    if visible is not None:
        if np.asarray(visible).shape != valid.shape:
            raise ValueError("Visibility must match the tracks' (T,K) shape.")
        valid &= np.asarray(visible).astype(bool)
    homogeneous = np.concatenate((safe, np.ones((*safe.shape[:-1], 1))), axis=-1)
    xyz = (homogeneous @ np.linalg.inv(intrinsics).T) * z[..., None]
    if camera_to_world is not None:
        transform = np.asarray(camera_to_world, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Camera-to-world transform must be a finite 4x4 matrix.")
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    return np.where(valid[..., None], xyz, 0).astype(np.float32), valid


def object_position_from_tracks(xyz, valid):
    """Visible-keypoint centroid and validity; coordinates share the point-cloud frame."""
    xyz, valid = np.asarray(xyz), np.asarray(valid).astype(bool)
    if xyz.shape[:-1] != valid.shape or xyz.shape[-1] != 3:
        raise ValueError("XYZ tracks and validity mask are inconsistent.")
    valid &= np.isfinite(xyz).all(-1)
    count = valid.sum(-1)
    position = np.where(valid[..., None], xyz, 0).sum(-2) / np.maximum(count[..., None], 1)
    return position.astype(np.float32), count > 0
