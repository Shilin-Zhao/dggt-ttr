import os
import time
import glob
import logging
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from dataclasses import dataclass, replace
from torch.utils.data import DataLoader
from torch.optim import Adam
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.losses import ssim_loss
from tqdm import tqdm

# Project Specific Imports
# 确保你的环境能找到这些包
from third_party.difix.infer import process_images_with_difix
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from dggt.models.vggt import VGGT
from sklearn.linear_model import RANSACRegressor
from vis_utils import compute_color_gradient_map

@dataclass
class DGGTConfig:
    # --- Basic Arguments ---
    image_dir: str
    ckpt_path: str
    output_path: str
    scene_names: list[str]
    
    input_views: int = 1
    sequence_length: int = 20
    start_idx: int = 0
    
    # --- Flags (对应命令行中的 -flag) ---
    images: bool = False        # 是否输出每一帧图像
    depth: bool = False         # 是否输出深度图 (.npy)
    metrics: bool = False       # 是否计算 PSNR/SSIM/LPIPS
    diffusion: bool = False     # 是否使用 Diffusion Refinement
    use_nearest:bool = False    # 使用nearest 对sky mask进行插值
    
    # --- Refinement Arguments (默认值保持与脚本一致) ---
    enable_refinement: bool = False
    enable_da3: bool = False    # Enable DA3 depth prior
    refine_pose: bool = False
    refine_gs: bool = False     # Opacity/Color/Rot
    refine_xyz: bool = False    # XYZ position
    
    refine_steps: int = 50      # Legacy total steps
    refine_steps_pose: int = 50
    refine_steps_gs: int = 50
    
    pose_lr_factor_gs: float = 0.1  # Scale pose LR during GS stage
    
    # --- Learning Rates ---
    penalize_alphas:bool = False
    refine_lr_pose: float = 1e-4
    refine_lr_opacity: float = 1e-3
    refine_lr_color: float = 1e-3
    refine_lr_rot: float = 1e-3
    refine_lr_scales: float = 3e-5
    refine_lr_xyz: float = 3e-5
    force_refresh: bool = False


def apply_diffusion(images, model_path, device):
    """Apply diffusion model to a batch of images."""
    processed_frames = []
    for i in range(images.shape[0]):
        frame = images[i].detach().cpu().clamp(0, 1)
        processed_frame = process_images_with_difix(frame, model_path)
        processed_frames.append(processed_frame)
    return torch.stack(processed_frames, dim=0).to(device)

def render_single_frame(world_points, rgbs, opacity, scales, rotation, 
                        extrinsic, intrinsic, H, W):
    """Render a single frame using Gaussian Splatting."""
    renders_chunk, alphas_chunk, _ = rasterization(
        means=world_points,
        quats=rotation,
        scales=scales,
        opacities=opacity,
        colors=rgbs,
        viewmats=extrinsic,
        Ks=intrinsic,
        width=W,
        height=H,
        render_mode='RGB+ED',
    )
    depth_chunk = renders_chunk[..., -1]
    rgb_chunk = renders_chunk[..., :-1]
    return rgb_chunk, alphas_chunk, depth_chunk

def render_all_frames(static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                      extrinsic, intrinsic, H, W, sky_model, images, penalize_alphas, sky_mask=None):
    """
    Render all frames for mode=2 (reconstruction).
    
    Returns:
        rendered_images: [T, C, H, W] rendered RGB images
        depth_maps: [T, H, W] depth maps
        alphas: [T, H, W, 1] alpha masks
    """
    num_frames = len(timestamps)
    chunked_renders, chunked_alphas = [], []
    
    for idx in range(num_frames):
        t0 = timestamps[idx]
        static_opacity_ = alpha_t(gs_timestamps, t0, static_gs['opacity'], gamma0=static_gs['gs_conf'])
        
        static_gs_list = [
            static_gs['points'], 
            static_gs['rgbs'], 
            static_opacity_, 
            static_gs['scales'], 
            static_gs['rotations']
        ]
        
        if dynamic_gs_list and len(dynamic_gs_list) > idx:
            dyn = dynamic_gs_list[idx]
            world_points, rgbs, opacity, scales, rotation = concat_list(
                static_gs_list,
                [dyn['points'], dyn['rgbs'], dyn['opacity'], dyn['scales'], dyn['rotations']]
            )
        else:
            world_points, rgbs, opacity, scales, rotation = static_gs_list
        
        rgb_chunk, alphas_chunk, depth_chunk = render_single_frame(
            world_points, rgbs, opacity, scales, rotation,
            extrinsic[idx:idx+1], intrinsic[idx:idx+1], H, W
        )
        renders_chunk = torch.cat([rgb_chunk, depth_chunk.unsqueeze(-1)], dim=-1)
        chunked_renders.append(renders_chunk)
        chunked_alphas.append(alphas_chunk)
    
    renders = torch.cat(chunked_renders, dim=0)
    depth_maps = renders[..., -1]
    renders = renders[..., :-1]
    alphas = torch.cat(chunked_alphas, dim=0)
    
    # Background blending
    bg_render = sky_model(images, extrinsic, intrinsic)
    bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)
    if penalize_alphas and sky_mask is not None:
        print("penalize_alphas, sky_mask shapes:", sky_mask.shape)
        alphas = alphas * (1-sky_mask[0][...,:1])
    renders = alphas * renders + (1 - alphas) * bg_render
    
    rendered_images = renders.permute(0, 3, 1, 2)
    return rendered_images, depth_maps, alphas


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    """Temporal opacity modulation."""
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()


def _concat_gaussian_chunks(chunks):
    """Concatenate gaussian chunks; keep schema even when empty."""
    if len(chunks) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return {
            "points": torch.empty(0, 3, dtype=torch.float32),
            "rgbs": torch.empty(0, 3, dtype=torch.float32),
            "opacity": empty,
            "scales": torch.empty(0, 3, dtype=torch.float32),
            "rotations": torch.empty(0, 4, dtype=torch.float32),
        }
    return {
        "points": torch.cat([c["points"] for c in chunks], dim=0),
        "rgbs": torch.cat([c["rgbs"] for c in chunks], dim=0),
        "opacity": torch.cat([c["opacity"] for c in chunks], dim=0),
        "scales": torch.cat([c["scales"] for c in chunks], dim=0),
        "rotations": torch.cat([c["rotations"] for c in chunks], dim=0),
    }


def build_gaussians_for_export(ply_payload, logical_frame_idx):
    """
    Build static/dynamic gaussian tensors for one logical frame.

    For views=3, a logical frame maps to 3 flattened indices and merges them.
    """
    views = int(ply_payload.get("views", 1))
    if views not in (1, 3):
        raise ValueError(f"Unsupported views={views}. Only 1 or 3 are supported.")

    timestamps_flat = ply_payload["timestamps_flat"]
    num_flat_frames = int(timestamps_flat.shape[0])
    if num_flat_frames % views != 0:
        raise ValueError(
            f"Inconsistent payload: num_flat_frames={num_flat_frames} is not divisible by views={views}."
        )
    num_logical_frames = num_flat_frames // views
    if logical_frame_idx < 0 or logical_frame_idx >= num_logical_frames:
        raise IndexError(
            f"logical_frame_idx={logical_frame_idx} out of range [0, {num_logical_frames - 1}]"
        )

    if views == 1:
        flat_indices = [logical_frame_idx]
    else:
        base = logical_frame_idx * views
        flat_indices = [base + i for i in range(views)]

    static_base = ply_payload["static_base"]
    dynamic_chunks = []
    dynamic_per_flat_frame = ply_payload["dynamic_per_flat_frame"]

    # Static should be exported once per logical frame.
    # For views=3, timestamps are repeated as [f0v0, f0v1, f0v2], so we use the first.
    t0 = timestamps_flat[flat_indices[0]]
    static_opacity_t = alpha_t(
        static_base["gs_timestamps"],
        t0,
        static_base["opacity"],
        gamma0=static_base["gs_conf"],
    )
    static_export = {
        "points": static_base["points"],
        "rgbs": static_base["rgbs"],
        "opacity": static_opacity_t,
        "scales": static_base["scales"],
        "rotations": static_base["rotations"],
    }

    for flat_idx in flat_indices:
        dynamic_chunks.append(dynamic_per_flat_frame[flat_idx])

    dynamic_export = _concat_gaussian_chunks(dynamic_chunks)
    return static_export, dynamic_export


def _write_gaussians_to_ply_ascii(gaussians, ply_path):
    """Write gaussian attributes to an ASCII PLY with custom properties."""
    points = gaussians["points"]
    rgbs = gaussians["rgbs"]
    opacity = gaussians["opacity"]
    scales = gaussians["scales"]
    rotations = gaussians["rotations"]

    num_points = int(points.shape[0])

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(rgbs, torch.Tensor):
        rgbs = rgbs.detach().cpu().numpy()
    if isinstance(opacity, torch.Tensor):
        opacity = opacity.detach().cpu().numpy()
    if isinstance(scales, torch.Tensor):
        scales = scales.detach().cpu().numpy()
    if isinstance(rotations, torch.Tensor):
        rotations = rotations.detach().cpu().numpy()

    rgbs = np.clip(rgbs, 0.0, 1.0).astype(np.float32)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    with open(ply_path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(num_points):
            x, y, z = points[i]
            f0, f1, f2 = rgbs[i]
            op = float(opacity[i])
            s0, s1, s2 = scales[i]
            q0, q1, q2, q3 = rotations[i]
            f.write(
                f"{x:.7f} {y:.7f} {z:.7f} "
                f"{f0:.7f} {f1:.7f} {f2:.7f} "
                f"{op:.7f} {s0:.7f} {s1:.7f} {s2:.7f} "
                f"{q0:.7f} {q1:.7f} {q2:.7f} {q3:.7f}\n"
            )


def _filter_gaussians_by_opacity(gaussians, min_opacity):
    """Filter gaussian attributes by opacity threshold."""
    opacity = gaussians["opacity"]
    threshold = float(min_opacity)
    if isinstance(opacity, torch.Tensor):
        mask = (opacity.reshape(-1) >= threshold)
    else:
        mask = (np.asarray(opacity).reshape(-1) >= threshold)

    filtered = {}
    for key, value in gaussians.items():
        if isinstance(value, torch.Tensor):
            value_mask = mask.to(device=value.device) if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).to(device=value.device)
            filtered[key] = value[value_mask]
        else:
            np_value = np.asarray(value)
            np_mask = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            filtered[key] = np_value[np_mask]
    return filtered


def save_gaussians_payload_to_ply(
    ply_payload,
    frame_indices,
    output_dir,
    scene_name,
    logger=None,
    min_opacity=0.0,
):
    """
    Export static/dynamic gaussians for selected logical frames.

    Args:
        min_opacity (float): keep gaussians with opacity >= min_opacity.

    Returns:
        List[dict]: file paths and point counts per frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported = []

    if frame_indices is None:
        raise ValueError("frame_indices must be provided, e.g. [0, 5].")
    if not isinstance(frame_indices, (list, tuple)):
        raise TypeError("frame_indices must be a list/tuple of logical frame indices.")
    if len(frame_indices) == 0:
        raise ValueError("frame_indices is empty.")
    if min_opacity is None:
        raise ValueError("min_opacity must be a number.")
    min_opacity = float(min_opacity)

    for logical_idx in frame_indices:
        static_export, dynamic_export = build_gaussians_for_export(ply_payload, int(logical_idx))
        static_export = _filter_gaussians_by_opacity(static_export, min_opacity)
        dynamic_export = _filter_gaussians_by_opacity(dynamic_export, min_opacity)

        static_path = os.path.join(output_dir, f"scene_{scene_name}_frame_{logical_idx}_static.ply")
        dynamic_path = os.path.join(output_dir, f"scene_{scene_name}_frame_{logical_idx}_dynamic.ply")

        _write_gaussians_to_ply_ascii(static_export, static_path)
        _write_gaussians_to_ply_ascii(dynamic_export, dynamic_path)

        record = {
            "frame_idx": int(logical_idx),
            "static_path": static_path,
            "dynamic_path": dynamic_path,
            "static_count": int(static_export["points"].shape[0]),
            "dynamic_count": int(dynamic_export["points"].shape[0]),
        }
        exported.append(record)
        if logger is not None:
            logger.info(
                f"Exported frame {logical_idx} (min_opacity={min_opacity}): "
                f"static={record['static_count']} -> {static_path}, "
                f"dynamic={record['dynamic_count']} -> {dynamic_path}"
            )
            if record["dynamic_count"] == 0:
                logger.warning(f"Frame {logical_idx} has empty dynamic gaussians.")
    return exported


# ========== Lie Algebra Utilities ==========

def se3_to_SE3(pose_6d):
    """
    Convert 6D pose (se3) to 4x4 transformation matrix (SE3).
    
    Args:
        pose_6d: [N, 6] tensor, first 3 are rotation (angle-axis), last 3 are translation
        
    Returns:
        SE3: [N, 4, 4] transformation matrices
    """
    N = pose_6d.shape[0]
    device = pose_6d.device
    
    angle_axis = pose_6d[:, :3]  # [N, 3]
    translation = pose_6d[:, 3:]  # [N, 3]
    
    R = axis_angle_to_rotation_matrix(angle_axis)  # [N, 3, 3]
    
    zeros = torch.zeros(N, 1, 3, device=device, dtype=pose_6d.dtype)
    ones = torch.ones(N, 1, 1, device=device, dtype=pose_6d.dtype)
    
    top_rows = torch.cat([R, translation.unsqueeze(-1)], dim=-1)  # [N, 3, 4]
    bottom_row = torch.cat([zeros, ones], dim=-1)  # [N, 1, 4]
    
    SE3 = torch.cat([top_rows, bottom_row], dim=1)  # [N, 4, 4]
    
    return SE3


def apply_pose_delta(extrinsic, pose_delta):
    """Apply pose delta to extrinsic matrix."""
    delta_SE3 = se3_to_SE3(pose_delta)
    new_extrinsic = torch.bmm(delta_SE3, extrinsic)
    return new_extrinsic


# ========== Refinement Trainer ==========

class RefinementTrainer:
    def __init__(self, config, logger, device='cuda'):
        self.config = config
        self.logger = logger
        self.device = device
        self.enable_refinement = config.enable_refinement
        self.refine_pose = config.refine_pose
        self.refine_gs = config.refine_gs
        self.refine_xyz = config.refine_xyz
        self.num_steps_pose = config.refine_steps_pose if config.refine_steps_pose is not None else config.refine_steps
        self.num_steps_gs = config.refine_steps_gs if config.refine_steps_gs is not None else config.refine_steps
        self.lr_pose = config.refine_lr_pose
        self.lr_opacity = config.refine_lr_opacity
        self.lr_color = config.refine_lr_color
        self.lr_rot = config.refine_lr_rot
        self.lr_scales = config.refine_lr_scales
        self.lr_xyz = config.refine_lr_xyz
        self.pose_lr_factor_gs = config.pose_lr_factor_gs
        
        if self.enable_refinement:
            self.logger.info(f"Refinement Trainer initialized.")
            self.logger.info(f"  - Refine Pose: {self.refine_pose} (LR: {self.lr_pose})")
            self.logger.info(f"  - Refine GS (opacity/color/rot/scales): {self.refine_gs} (LRs: opacity {self.lr_opacity}, color {self.lr_color}, rot {self.lr_rot}, scales {self.lr_scales})")
            self.logger.info(f"  - Refine XYZ: {self.refine_xyz} (LR: {self.lr_xyz}, only if GS enabled)")
            self.logger.info(f"  - Steps: pose {self.num_steps_pose}, gs {self.num_steps_gs}, pose_lr_factor_gs {self.pose_lr_factor_gs}")

    def compute_loss(self, pred_image, gt_image, mask=None, skip_first_frame=False):
        """
        Compute RGB L1 + SSIM loss with optional masking.
        
        Args:
            pred_image: [T, C, H, W] predicted images
            gt_image: [T, C, H, W] ground truth images
            mask: optional [T, H, W] mask (1 = valid, 0 = ignore)
            skip_first_frame: if True, exclude first frame from loss calculation
        """
        # Skip first frame to avoid overfitting
        if skip_first_frame and pred_image.shape[0] > 1:
            pred_image = pred_image[1:]
            gt_image = gt_image[1:]
            if mask is not None:
                mask = mask[1:]
        
        if mask is not None:
            m = mask.unsqueeze(1).float()  # [T, 1, H, W]
            valid_pixels = m.sum() + 1e-8
            
            # L1 loss: 先算差异，再 mask
            l1_loss = (torch.abs(pred_image - gt_image) * m).sum() / (valid_pixels * pred_image.shape[1])
            
            # SSIM loss: 先全图计算 loss map，再 mask（避免破坏边界统计特征）
            ssim_map = ssim_loss(pred_image, gt_image, window_size=11, reduction='none')  # [T, C, H, W]
            ssim_val = (ssim_map * m).sum() / (valid_pixels * pred_image.shape[1])
        else:
            l1_loss = F.l1_loss(pred_image, gt_image, reduction='mean')
            ssim_val = ssim_loss(pred_image, gt_image, window_size=11)
        
        total_loss = l1_loss + 0.2 * ssim_val
        return total_loss, l1_loss, ssim_val

    def render_with_pose(self, static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                         extrinsic, intrinsic, H, W, sky_model, images):
        """Render all frames with given pose (supports gradient)."""
        num_frames = len(timestamps)
        chunked_renders, chunked_alphas = [], []
        
        for idx in range(num_frames):
            t0 = timestamps[idx]
            static_opacity_ = alpha_t(gs_timestamps, t0, static_gs['opacity'], gamma0=static_gs['gs_conf'])
            
            static_gs_list = [
                static_gs['points'], 
                static_gs['rgbs'], 
                static_opacity_, 
                static_gs['scales'], 
                static_gs['rotations']
            ]
            
            if dynamic_gs_list and len(dynamic_gs_list) > idx:
                dyn = dynamic_gs_list[idx]
                world_points, rgbs, opacity, scales, rotation = concat_list(
                    static_gs_list,
                    [dyn['points'], dyn['rgbs'], dyn['opacity'], dyn['scales'], dyn['rotations']]
                )
            else:
                world_points, rgbs, opacity, scales, rotation = static_gs_list
            
            renders_chunk, alphas_chunk, _ = rasterization(
                means=world_points,
                quats=rotation,
                scales=scales,
                opacities=opacity,
                colors=rgbs,
                viewmats=extrinsic[idx:idx+1],
                Ks=intrinsic[idx:idx+1],
                width=W,
                height=H,
                render_mode='RGB+ED',
            )
            chunked_renders.append(renders_chunk)
            chunked_alphas.append(alphas_chunk)
        
        renders = torch.cat(chunked_renders, dim=0)
        depth_maps = renders[..., -1]
        renders = renders[..., :-1]
        alphas = torch.cat(chunked_alphas, dim=0)
        
        with torch.no_grad():
            bg_render = sky_model(images, extrinsic, intrinsic)
            bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)
        
        renders = alphas * renders + (1 - alphas) * bg_render
        rendered_images = renders.permute(0, 3, 1, 2)
        
        return rendered_images, depth_maps, alphas

    def optimize_stage(self, gt_images, static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                      extrinsic, intrinsic, H, W, sky_model, images,
                      refine_pose_flag=True, refine_gs_flag=False, refine_xyz_flag=False,
                      lr_pose_override=None, num_steps=50, loss_mask=None):
        """Optimize camera poses and optional GS outputs for one stage."""
        num_frames = extrinsic.shape[0]
        trainable_params = []
        
        pose_delta = torch.zeros(num_frames, 6, device=self.device, requires_grad=False)
        pose_delta_trainable = None
        
        if refine_pose_flag:
            lr_pose = self.lr_pose if lr_pose_override is None else lr_pose_override
            if num_frames > 1:
                pose_delta_trainable = torch.zeros(num_frames - 1, 6, device=self.device, requires_grad=True)
                trainable_params.append({'params': [pose_delta_trainable], 'lr': lr_pose})
                self.logger.info(f"Pose optimization: fixing frame 0, optimizing frames 1-{num_frames-1}")
            else:
                self.logger.warning("Only 1 frame, skipping pose optimization")
        
        current_static_gs = {}
        for k, v in static_gs.items():
            if isinstance(v, torch.Tensor):
                current_static_gs[k] = v.detach().clone()
            else:
                current_static_gs[k] = v
        
        if refine_gs_flag:
            if 'opacity' in current_static_gs:
                current_static_gs['opacity'].requires_grad_(True)
                trainable_params.append({'params': [current_static_gs['opacity']], 'lr': self.lr_opacity})
            if 'rgbs' in current_static_gs:
                current_static_gs['rgbs'].requires_grad_(True)
                trainable_params.append({'params': [current_static_gs['rgbs']], 'lr': self.lr_color})
            if 'rotations' in current_static_gs:
                current_static_gs['rotations'].requires_grad_(True)
                trainable_params.append({'params': [current_static_gs['rotations']], 'lr': self.lr_rot})
            if 'scales' in current_static_gs and self.lr_scales > 0.:
                current_static_gs['scales'].requires_grad_(True)
                trainable_params.append({'params': [current_static_gs['scales']], 'lr': self.lr_scales})
            if refine_xyz_flag and 'points' in current_static_gs:
                current_static_gs['points'].requires_grad_(True)
                trainable_params.append({'params': [current_static_gs['points']], 'lr': self.lr_xyz})
        
        for dyn in dynamic_gs_list:
            for key in dyn:
                if isinstance(dyn[key], torch.Tensor):
                    dyn[key] = dyn[key].detach()
        
        if not trainable_params:
            return extrinsic.detach(), static_gs
        
        optimizer = Adam(trainable_params, betas=(0.9, 0.999))
        self.logger.info(f"Starting Optimization for {num_steps} steps... (pose={refine_pose_flag}, gs={refine_gs_flag}, xyz={refine_xyz_flag})")
        
        best_loss = float('inf')
        best_pose_delta = torch.zeros(num_frames, 6, device=self.device)
        best_static = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in current_static_gs.items()}
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            if pose_delta_trainable is not None:
                pose_delta = torch.cat([
                    torch.zeros(1, 6, device=self.device),
                    pose_delta_trainable
                ], dim=0)
            
            current_extrinsic = apply_pose_delta(extrinsic.detach(), pose_delta)
            
            rendered_images, _, alphas = self.render_with_pose(
                current_static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                current_extrinsic, intrinsic, H, W, sky_model, images
            )
            
            total_loss, l1_loss, ssim_val = self.compute_loss(rendered_images, gt_images, mask=loss_mask, skip_first_frame=True)
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_pose_delta = pose_delta.clone().detach()
                best_static = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in current_static_gs.items()}
            
            if step % 10 == 0 or step == num_steps - 1:
                self.logger.info(f"  Step {step:3d}/{num_steps}: Loss={total_loss.item():.6f} (L1={l1_loss.item():.6f}, SSIM={ssim_val.item():.6f})")
        
        optimized_extrinsic = apply_pose_delta(extrinsic.detach(), best_pose_delta)
        self.logger.info(f"Optimization finished. Best Loss: {best_loss:.6f}")
        return optimized_extrinsic.detach(), best_static

    def forward(self, gt_images, static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                extrinsic, intrinsic, H, W, sky_model, images, sky_mask=None, dynamic_mask=None):
        """Main entry point for refinement."""
        if not self.enable_refinement:
            return extrinsic, static_gs
        
        torch.cuda.empty_cache()
        
        loss_mask = None
        if sky_mask is not None:
            if sky_mask.dim() == 4:
                loss_mask = (sky_mask == 0).any(dim=-1).float()
            else:
                loss_mask = (sky_mask == 0).float()
            self.logger.info(f"Using sky mask for loss computation. Valid pixels: {loss_mask.sum().item():.0f}/{loss_mask.numel()}")
        
        optimized_extrinsic = extrinsic
        optimized_static_gs = static_gs
        
        # Phase 1: Pose-only
        if self.refine_pose:
            self.logger.info("=== Phase 1: Pose Optimization ===")
            optimized_extrinsic, optimized_static_gs = self.optimize_stage(
                gt_images, static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                optimized_extrinsic, intrinsic, H, W, sky_model, images,
                refine_pose_flag=True, refine_gs_flag=False, refine_xyz_flag=False,
                lr_pose_override=self.lr_pose, num_steps=self.num_steps_pose,
                loss_mask=loss_mask
            )
        
        # Phase 2: GS (and optional low-lr pose)
        if self.refine_gs:
            self.logger.info("=== Phase 2: GS Optimization (opacity/color/rot + optional xyz) ===")
            pose_lr_stage2 = self.lr_pose * self.pose_lr_factor_gs if self.refine_pose else None
            refine_pose_stage2 = self.refine_pose and self.pose_lr_factor_gs > 0
            optimized_extrinsic, optimized_static_gs = self.optimize_stage(
                gt_images, optimized_static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                optimized_extrinsic, intrinsic, H, W, sky_model, images,
                refine_pose_flag=refine_pose_stage2,
                refine_gs_flag=True,
                refine_xyz_flag=self.refine_xyz,
                lr_pose_override=pose_lr_stage2,
                num_steps=self.num_steps_gs,
                loss_mask=loss_mask
            )
        
        return optimized_extrinsic, optimized_static_gs


# -----------------------------------------------------------------------------
# 3. Inference & Rendering Pipeline
# -----------------------------------------------------------------------------


# --- Cache Utilities (辅助函数) ---
def get_baseline_cache_dir(scene_name, start_idx, sequence_length):
    """Get the baseline cache directory path."""
    # 这里的路径可以根据你的实际需求调整，目前保持和脚本一致
    cache_dir = f"./outputs/baseline/{scene_name}/{start_idx}_{sequence_length}"
    return cache_dir

def save_predictions_cache(cache_dir, predictions, logger):
    """Save model predictions to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "predictions.pt")
    
    cache_data = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            cache_data[key] = value.detach().cpu()
        else:
            cache_data[key] = value
    
    torch.save(cache_data, cache_path)
    logger.info(f"Saved predictions cache to {cache_path}")

def load_predictions_cache(cache_dir, device, logger):
    """Load model predictions from cache."""
    cache_path = os.path.join(cache_dir, "predictions.pt")
    if not os.path.exists(cache_path):
        return None
    
    cache_data = torch.load(cache_path, map_location=device)
    logger.info(f"Loaded predictions cache from {cache_path}")
    return cache_data

def compute_metrics(img1, img2, loss_fn):
    """Compute PSNR, SSIM, and LPIPS metrics between two images."""
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1, img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(lpips_list) / len(lpips_list)

# --- Main Pipeline Function ---
def run_inference_and_render(config, model, dataset, logger, device='cuda', return_ply=False):
    """
    Run the full DGGT inference and rendering pipeline.
    Returns a list of dictionaries containing results for each scene in the dataset.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    
    dtype = torch.float32
    
    # Initialize metrics computation if enabled
    loss_fn = None
    psnr_list, ssim_list, lpips_list = [], [], []
    if config.metrics:
        loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Initialize Refinement Trainer if enabled
    refiner = None
    if config.enable_refinement:
        refiner = RefinementTrainer(config, logger, device)
    
    # 获取场景列表，用于缓存命名
    # 注意：Config 中的 scene_names 是一个 list
    scene_names = config.scene_names
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 1. Prepare Data
            images = batch['images'].to(device)
            # Sky mask: [B, S, C, H, W] -> permute -> [B, S, H, W, C]
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            nearest_sky_mask = batch['nearest_masks'].to(device).permute(0, 1, 3, 4, 2)
            if config.use_nearest:
                logger.info("use nearest sky mask")
                sky_mask = nearest_sky_mask
            gt_dy_map = batch['dynamic_mask'].to(device) if 'dynamic_mask' in batch else None
            gt_depth = batch['gt_depth'].to(device) if 'gt_depth' in batch else None
            
            # Load DA3 data if available
            da3_depth = batch['da3_depth'].to(device) if 'da3_depth' in batch else None
            da3_sky_mask = batch['da3_sky_mask'].to(device) if 'da3_sky_mask' in batch else None

            # Background mask logic (user's logic)
            # mask == 0 implies Sky in this logic, so bg_mask=True means Sky? 
            # Wait, usually bg_mask means "Valid Background Region" (not sky) or "Sky Region"?
            # Let's check original code: static_mask = (bg_mask & (dy_map < 0.5))
            # If bg_mask is Sky, then static_mask would be Sky & Static... which implies static Gaussians are placed in the Sky?
            # Actually usually: sky_mask==0 means VALID PIXELS (not sky).
            # Let's assume: bg_mask = True for VALID regions (buildings, road), False for Sky.
            bg_mask = (sky_mask == 0).any(dim=-1) 
            
            timestamps = batch['timestamps'][0].to(device)
            start_time = time.time()

            # 2. Cache Handling
            # Determine scene name safely
            if batch_idx < len(scene_names):
                scene_name = scene_names[batch_idx]
            else:
                scene_name = str(batch_idx).zfill(3)

            predictions = model(images)

            # 3. Post-Process Predictions -> Gaussians
            with torch.cuda.amp.autocast(dtype=dtype):
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                # Add [0,0,0,1] row to make 4x4
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                # Depth to Point Cloud
                use_depth = True
                if use_depth:
                    depth_map = predictions["depth"][0]  # [T, H, W, 1]
                    # print("depth_map.shape:", depth_map.shape)
                else:
                    depth_map = None
                    
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1)
                
                # DA3 Processing (if enabled, directly overwrite depth_map and sky_mask)
                if config.enable_da3 and da3_depth is not None and da3_sky_mask is not None:
                    logger.info("Processing DA3 depth prior...")
                    T = depth_map.shape[0]
                    depth_final_list = []
                    sky_final_list = []
                    
                    # Process each frame
                    for t in range(T):
                        # Convert to numpy
                        d_dggt = depth_map[t].detach().cpu().numpy()
                        d_da3 = da3_depth[0, t, 0].detach().cpu().numpy()  # [B, T, 1, H, W] -> [H, W]
                        
                        # Process nearest_sky_mask: [B, T, H, W, C] -> [H, W] (1.0=Sky, 0.0=Obj)
                        # Same logic as extract_depth_alignment_data
                        m_dggt_bool = (nearest_sky_mask[0, t] != 0).any(dim=-1)  # [H, W] Boolean
                        m_dggt = m_dggt_bool.float().detach().cpu().numpy()  # [H, W] float (1.0=Sky, 0.0=Obj)
                        
                        m_da3 = da3_sky_mask[0, t, 0].detach().cpu().numpy()  # [B, T, 1, H, W] -> [H, W]
                        
                        # Process single frame
                        img_np = images[0, t].detach().cpu().permute(1, 2, 0).numpy()
                        d_final, m_final, _ = process_depth_fusion_single_frame(
                            d_dggt, d_da3, m_dggt, m_da3,
                            image_np=img_np,
                            logger=logger
                        )
                        
                        depth_final_list.append(d_final)
                        sky_final_list.append(m_final)
                    
                    # Stack results: [T, H, W, 1]
                    depth_map = torch.from_numpy(np.stack(depth_final_list, axis=0)).to(device).float().unsqueeze(-1)
                    sky_final = np.stack(sky_final_list, axis=0)  # [T, H, W], 1=Sky, 0=Object
                    
                    # Convert sky_final to sky_mask format: [B, T, H, W, C]
                    # sky_final: [T, H, W], 1=Sky, 0=Object
                    # sky_mask: [B, T, H, W, C], non-zero=Sky
                    sky_final_tensor = torch.from_numpy(sky_final).to(device).float()  # [T, H, W]
                    sky_final_expanded = sky_final_tensor.unsqueeze(0).unsqueeze(-1)  # [1, T, H, W, 1]
                    C = sky_mask.shape[-1]
                    sky_mask = sky_final_expanded.expand(1, T, sky_final.shape[1], sky_final.shape[2], C)
                    
                    logger.info("DA3 processing complete, depth_map and sky_mask updated")
                
                # Convert depth_map to point_map
                if use_depth:
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:
                    point_map = predictions["world_points"]
                
                # Recompute bg_mask after potential DA3 processing
                bg_mask = (sky_mask == 0).any(dim=-1)

                # --- Split Static / Dynamic ---
                # Static Mask: Valid Background AND Not Dynamic
                static_mask = (bg_mask & (dy_map < 0.5))
                
                static_points = point_map[static_mask].reshape(-1, 3)
                gs_dynamic_list = dy_map[static_mask].sigmoid()
                
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                static_opacity = static_opacity * (1 - gs_dynamic_list)
                static_gs_conf = gs_conf[static_mask]
                
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                gs_timestamps = timestamps[frame_idx]

                # Pack Static Dict
                static_gs = {
                    'points': static_points,
                    'rgbs': static_rgbs,
                    'opacity': static_opacity,
                    'scales': static_scales,
                    'rotations': static_rotations,
                    'gs_conf': static_gs_conf
                }
                
                # Pack Dynamic List
                dynamic_gs_list = []
                for i in range(dy_map.shape[1]): # Iterate frames
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    
                    # Dynamic Gaussians logic
                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                    
                    dynamic_gs_list.append({
                        'points': dynamic_point,
                        'rgbs': dynamic_rgb,
                        'opacity': dynamic_opacity,
                        'scales': dynamic_scale,
                        'rotations': dynamic_rotation
                    })
                
                target_image = images[0]

            # 4. Render Initial Result
            logger.info("Rendering frames...")
            rendered_image, depth_maps, alphas = render_all_frames(
                static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                extrinsic, intrinsic, H, W, model.sky_model, images, config.penalize_alphas, sky_mask=sky_mask
            )

            # 5. Refinement Logic (if enabled, directly overwrite variables)
            if config.enable_refinement and refiner is not None:
                logger.info("Starting Test-Time Refinement...")
                
                with torch.enable_grad():
                    extrinsic, static_gs = refiner.forward(
                        gt_images=target_image,
                        static_gs=static_gs,
                        dynamic_gs_list=dynamic_gs_list,
                        timestamps=timestamps,
                        gs_timestamps=gs_timestamps,
                        extrinsic=extrinsic,
                        intrinsic=intrinsic,
                        H=H, W=W,
                        sky_model=model.sky_model,
                        images=images,
                        sky_mask=sky_mask[0],
                        dynamic_mask=gt_dy_map[0] if gt_dy_map is not None else None
                    )
                
                # Re-render with optimized results
                logger.info("Rendering final frames (after refinement)...")
                with torch.no_grad():
                    rendered_image, depth_maps, alphas = render_all_frames(
                        static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                        extrinsic, intrinsic, H, W, model.sky_model, images, config.penalize_alphas, sky_mask=sky_mask
                    )
            
            # 6. Apply Diffusion (if enabled, directly overwrite rendered_image)
            if config.diffusion:
                logger.info("Applying diffusion")
                rendered_image = apply_diffusion(rendered_image, "/root/autodl-tmp/dggt-ttr/pretrained/diffusion_model.pth", device)
                
            # 7. Compute Metrics (if enabled)
            metrics = None
            if config.metrics and loss_fn is not None:
                psnr, ssim, lpip = compute_metrics(rendered_image, target_image, loss_fn)
                metrics = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpip
                }
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpip)
                logger.info(f"Scene {scene_name} Metrics: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpip:.4f}")
            
            # 8. Collect Results (无论是否计算 metrics 都需要收集结果)
            scene_result = {
                'scene_name': scene_name,
                'rendered_image': rendered_image.detach().cpu(),  # [T, C, H, W]
                'depth_maps': depth_maps.detach().cpu(),          # [T, H, W]
                'alphas': alphas.detach().cpu(),                  # [T, H, W, 1]
                'gt_image': target_image.detach().cpu(),          # [T, C, H, W]
                'gt_dynamic_mask': gt_dy_map.detach().cpu() if gt_dy_map is not None else None,
                'masks': batch['masks'].cpu(), # Keep original masks for vis
                'nearest_masks': batch['nearest_masks'].cpu()
            }
            if return_ply:
                dynamic_per_flat_frame = []
                for dyn in dynamic_gs_list:
                    dynamic_per_flat_frame.append({
                        'points': dyn['points'].detach().cpu(),
                        'rgbs': dyn['rgbs'].detach().cpu(),
                        'opacity': dyn['opacity'].detach().cpu(),
                        'scales': dyn['scales'].detach().cpu(),
                        'rotations': dyn['rotations'].detach().cpu(),
                    })
                scene_result['ply_payload'] = {
                    'views': int(getattr(config, "input_views", 1)),
                    'timestamps_flat': timestamps.detach().cpu(),
                    'dynamic_per_flat_frame': dynamic_per_flat_frame,
                    'static_base': {
                        'points': static_gs['points'].detach().cpu(),
                        'rgbs': static_gs['rgbs'].detach().cpu(),
                        'opacity': static_gs['opacity'].detach().cpu(),
                        'scales': static_gs['scales'].detach().cpu(),
                        'rotations': static_gs['rotations'].detach().cpu(),
                        'gs_conf': static_gs['gs_conf'].detach().cpu(),
                        'gs_timestamps': gs_timestamps.detach().cpu(),
                    },
                }
            if da3_depth is not None:
                scene_result['da3_depth'] = da3_depth.detach().cpu()
            if metrics is not None:
                scene_result['metrics'] = metrics
            
            results.append(scene_result)
            logger.info(f"Scene {scene_name} processed. Inference time: {time.time() - start_time:.2f}s")
    
    # 8. Log Final Average Metrics (if enabled)
    if config.metrics and loss_fn is not None and len(psnr_list) > 0:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_lpips = sum(lpips_list) / len(lpips_list)
        logger.info(f"Final Average Metrics across all scenes:")
        logger.info(f"  PSNR: {avg_psnr:.4f}")
        logger.info(f"  SSIM: {avg_ssim:.4f}")
        logger.info(f"  LPIPS: {avg_lpips:.4f}")
            
    return results

def load_model(config, device, logger):
    model = VGGT().to(device)
    # Load Checkpoint
    checkpoint = torch.load(config.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    logger.info(f"Model loaded from {config.ckpt_path} successfully.")
    return model



def get_dggt_raw_depth(config, model, dataset, device='cuda'):
    """
    Helper: Run model inference to get raw predicted depth maps without rendering.
    Returns:
        dggt_depths: [T, H, W] tensor
        batch_data: dict containing original batch data for masks etc.
    """
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Take the first batch (sequence)
    batch = next(iter(dataloader))
    
    images = batch['images'].to(device)
    
    with torch.no_grad():
        predictions = model(images)
        
    # DGGT usually outputs depth in predictions['depth'] or inside 'world_points'
    # Assuming predictions['depth'] exists and is [B, T, H, W] or [1, T, H, W]
    if 'depth' in predictions:
        dggt_depth = predictions['depth'][0] # [T, H, W]
    else:
        # Fallback: project world points to depth if explicit depth head missing
        # This depends on your specific DGGT implementation details
        raise NotImplementedError("Model output dictionary must contain 'depth' key.")
        
    return dggt_depth, batch

def align_depth_disparity(dggt_depth, da3_depth, mask_valid):
    """
    Aligns DA3 depth to DGGT depth using Disparity (1/depth).
    Performs RANSAC on: 1/d_dggt = s * (1/d_da3) + t
    
    Args:
        dggt_depth: [H, W] numpy
        da3_depth: [H, W] numpy
        mask_valid: [H, W] boolean numpy (Valid intersection regions)
        
    Returns:
        da3_depth_aligned: [H, W] numpy
        scale, shift: floats
    """
    # Avoid division by zero
    eps = 1e-6
    
    # 1. Convert to Disparity
    disp_dggt = 1.0 / (dggt_depth + eps)
    disp_da3 = 1.0 / (da3_depth + eps)
    
    # 2. Select valid points
    y = disp_dggt[mask_valid].flatten()
    X = disp_da3[mask_valid].flatten().reshape(-1, 1)
    
    if len(y) < 100:
        print("[Warning] Not enough points for RANSAC, return original.")
        return da3_depth, 1.0, 0.0

    # 4. RANSAC Fitting
    ransac = RANSACRegressor(min_samples=0.5, residual_threshold=0.1, random_state=42)
    ransac.fit(X, y)
    
    scale = ransac.estimator_.coef_[0]
    shift = ransac.estimator_.intercept_
    
    # 5. Apply Transform to Full DA3 Disparity
    # aligned_disp = s * disp_da3 + t
    disp_da3_aligned = scale * disp_da3 + shift
    
    # 6. Convert back to Depth
    # depth = 1 / disp.  Handle negative disparity issues (clamp to min positive)
    disp_da3_aligned = np.maximum(disp_da3_aligned, 1e-4) # Avoid negative depth
    depth_da3_aligned = 1.0 / disp_da3_aligned
    
    return depth_da3_aligned, scale, shift



def run_depth_alignment_debug(
    sample_indices, 
    dggt_depth_seq, dggt_sky_seq, 
    da3_depth_seq, da3_sky_seq,
    dataset=None,
    logger=None, visualize_func=None
):
    """
    执行核心的 Debug 循环：数据转 Numpy -> 调用融合算法 -> 调用可视化
    """
    seq_len = dggt_depth_seq.shape[0]
    
    # 获取图像数据序列 (如果有传 dataset)
    image_seq = None
    if dataset is not None:
        batch_data = dataset[0] # 取第一批数据
        image_seq = batch_data['images'] # [S, C, H, W]

    for t in sample_indices:
        if t >= seq_len: 
            if logger: logger.info(f"Frame {t} out of bounds, skipping.")
            continue
        
        if logger: logger.info(f"--- Processing Depth Alignment Frame {t} ---")
        
        # 1. Tensor -> Numpy
        d_dggt = dggt_depth_seq[t].detach().cpu().numpy()
        d_da3 = da3_depth_seq[t].detach().cpu().numpy()
        m_dggt = dggt_sky_seq[t].detach().cpu().numpy()
        m_da3 = da3_sky_seq[t].detach().cpu().numpy()
        
        # 2. 核心融合逻辑 (调用你之前定义的 process_depth_fusion_single_frame)
        img_np = None
        if image_seq is not None:
            img_t = image_seq[t] # [C, H, W]
            if hasattr(img_t, "detach"):
                # 转成 [H, W, C] 格式
                img_np = img_t.detach().cpu().permute(1, 2, 0).numpy()
            else:
                img_np = np.asarray(img_t)

        d_final, m_final, d_da3_aligned = process_depth_fusion_single_frame(
            d_dggt, d_da3, m_dggt, m_da3,
            image_np=img_np,
            logger=logger
        )
        
        # 3. 可视化
        if visualize_func:
            visualize_func(
                t,
                d_dggt, m_dggt,
                d_da3_aligned, m_da3, 
                d_final, m_final
            )


# --- core_pipeline.py 修正版 ---

def extract_depth_alignment_data(config, model, dataset, device='cuda'):
    """
    运行模型推理并提取所有对齐所需的原始 Tensor 数据。
    修正了维度提取逻辑和天空/物品的语义。
    """
    # 1. 获取 DGGT 原始深度
    dggt_depth_tensor, batch_data = get_dggt_raw_depth(config, model, dataset, device)
    
    # --- 修正 1: DGGT Mask 处理 ---
    # 原始: [B, S, C, H, W] -> Permute -> [B, S, H, W, C]
    nearest_masks = batch_data['nearest_masks'].to(device)
    nearest_masks = nearest_masks.permute(0, 1, 3, 4, 2)
    
    # 语义修正: mask != 0 代表天空 (Sky)，== 0 代表物品 (Object)
    # [B, S, H, W, C] -> [B, S, H, W] (Boolean)
    # 这里得到的是 Sky Mask (True = Sky)
    dggt_sky_seq_bool = (nearest_masks != 0).any(dim=-1)
    
    # 取 Batch 0 -> [S, H, W]
    dggt_sky_seq = dggt_sky_seq_bool[0].float() # 转为 float 方便后续处理 (1.0=Sky, 0.0=Obj)

    # --- 修正 2: DGGT Depth 维度 ---
    # [T, 1, H, W] -> [T, H, W]
    if dggt_depth_tensor.ndim == 4:
        dggt_depth_tensor = dggt_depth_tensor.squeeze(1)
    
    # --- 修正 3: DA3 数据处理 ---
    # DA3 Depth: batch['da3_depth'] 通常是 [B, T, 1, H, W]
    da3_depth_seq = batch_data['da3_depth'][0] # -> [T, 1, H, W]
    if da3_depth_seq.ndim == 4:
        da3_depth_seq = da3_depth_seq.squeeze(1) # -> [T, H, W]
    da3_depth_seq = da3_depth_seq.to(device)

    # DA3 Sky Mask: batch['da3_sky_mask'] 通常是 [B, T, 1, H, W]
    da3_sky_seq = batch_data['da3_sky_mask'][0] # -> [T, 1, H, W]
    if da3_sky_seq.ndim == 4:
        da3_sky_seq = da3_sky_seq.squeeze(1) # -> [T, H, W]
    da3_sky_seq = da3_sky_seq.to(device)
    
    return dggt_depth_tensor, dggt_sky_seq, da3_depth_seq, da3_sky_seq


def compute_color_gradient_magnitude(image_np):
    """
    统一复用 vis_utils.py 中的梯度实现，避免调试图和实际融合逻辑不一致。
    """
    if image_np is None:
        return None

    img = image_np
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = img[..., None]

    if img.ndim != 3:
        return None

    return compute_color_gradient_map(img)


def extract_top_connected_sky_seed(sure_sky_mask):
    """
    仅提取与图像上边界连通的 sure_sky，作为天空扩张种子。
    """
    sure_sky_mask = np.asarray(sure_sky_mask, dtype=bool)
    if sure_sky_mask.ndim != 2 or not sure_sky_mask.any():
        return np.zeros_like(sure_sky_mask, dtype=bool)

    h, w = sure_sky_mask.shape
    seed_mask = np.zeros((h, w), dtype=bool)
    queue = deque()

    top_cols = np.where(sure_sky_mask[0])[0]
    for x in top_cols:
        seed_mask[0, x] = True
        queue.append((0, x))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and sure_sky_mask[ny, nx] and not seed_mask[ny, nx]:
                seed_mask[ny, nx] = True
                queue.append((ny, nx))

    return seed_mask


def expand_sky_through_low_gradient(top_sky_seed, unknown_mask, grad_mag, grad_thresh):
    """
    天空从外往内扩张：
    只允许从 top-connected sky seed 穿过低梯度的 unknown 区域。
    未被扩张到的 unknown 将在后续全部视为物体。
    """
    top_sky_seed = np.asarray(top_sky_seed, dtype=bool)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    if not top_sky_seed.any() or not unknown_mask.any():
        return np.zeros_like(unknown_mask, dtype=bool)

    h, w = unknown_mask.shape
    expanded_sky = np.zeros((h, w), dtype=bool)
    visited = top_sky_seed.copy()
    queue = deque([tuple(idx) for idx in np.argwhere(top_sky_seed)])
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if visited[ny, nx]:
                continue
            if unknown_mask[ny, nx] and grad_mag[ny, nx] <= grad_thresh:
                expanded_sky[ny, nx] = True
                visited[ny, nx] = True
                queue.append((ny, nx))
            elif unknown_mask[ny, nx]:
                visited[ny, nx] = True
            elif not unknown_mask[ny, nx]:
                visited[ny, nx] = True

    return expanded_sky


def build_edge_barrier(unknown_mask, grad_mag, grad_thresh, dilate_kernel_size=3):
    """
    在争议区内构造高梯度屏障，避免天空从电线杆边缘的缝隙漏进来。
    """
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    if not unknown_mask.any():
        return np.zeros_like(unknown_mask, dtype=bool)

    edge_barrier = unknown_mask & (grad_mag >= grad_thresh)
    if edge_barrier.any() and dilate_kernel_size > 1:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
        edge_barrier = cv2.dilate(edge_barrier.astype(np.uint8), kernel, iterations=1).astype(bool)
        edge_barrier = edge_barrier & unknown_mask

    return edge_barrier


def process_depth_fusion_single_frame(
    dggt_depth_np, da3_depth_np, 
    nearest_sky_mask_np, da3_sky_mask_np,
    image_np=None,
    logger=None
):
    """
    单帧融合逻辑。
    """
    # 1. 维度安全检查: 强制压缩成 [H, W]
    if dggt_depth_np.ndim == 3: dggt_depth_np = dggt_depth_np.squeeze()
    if da3_depth_np.ndim == 3: da3_depth_np = da3_depth_np.squeeze()
    if nearest_sky_mask_np.ndim == 3: nearest_sky_mask_np = nearest_sky_mask_np.squeeze()
    if da3_sky_mask_np.ndim == 3: da3_sky_mask_np = da3_sky_mask_np.squeeze()

    H, W = dggt_depth_np.shape

    # 2. 定义 Object Mask (非天空)
    # 输入语义: >0 (或 True) 是天空, 0 (或 False) 是物品
    is_dggt_obj = (nearest_sky_mask_np == 0) # True = Object
    is_da3_obj = (da3_sky_mask_np == 0)      # True = Object
    
    sure_obj = is_dggt_obj & is_da3_obj      # 双方都认为是物体
    sure_sky = (~is_dggt_obj) & (~is_da3_obj) # 双方都认为是天空
    unknown_mask = is_dggt_obj ^ is_da3_obj   # 有分歧的区域
    
    # 3. 对齐 (Alignment)
    da3_aligned, s, t = align_depth_disparity(dggt_depth_np, da3_depth_np, sure_obj)
    
    if logger:
        logger.info(f"  Alignment: Scale={s:.4f}, Shift={t:.4f}")

    # 4. 融合 Sky Mask：
    # sure_sky / sure_obj 保持不变；
    # unknown 仅允许从上边界连通的天空种子穿过低梯度区域向内扩张；
    # 扩张不到的 unknown 统一视为物体。
    expanded_sky = np.zeros_like(sure_sky, dtype=bool)
    grad_thresh = 0.12
    if image_np is not None and unknown_mask.any():
        grad_mag = compute_color_gradient_magnitude(image_np)
        if grad_mag is not None:
            top_sky_seed = extract_top_connected_sky_seed(sure_sky)
            edge_barrier = build_edge_barrier(
                unknown_mask=unknown_mask,
                grad_mag=grad_mag,
                grad_thresh=grad_thresh,
                dilate_kernel_size=3
            )
            passable_unknown = unknown_mask & (~edge_barrier)
            expanded_sky = expand_sky_through_low_gradient(
                top_sky_seed=top_sky_seed,
                unknown_mask=passable_unknown,
                grad_mag=grad_mag,
                grad_thresh=grad_thresh
            )
            if logger:
                logger.info(
                    f"  Unknown refined by sky expansion: "
                    f"unknown={int(unknown_mask.sum())}, "
                    f"top_seed={int(top_sky_seed.sum())}, "
                    f"edge_barrier={int(edge_barrier.sum())}, "
                    f"expanded_sky={int(expanded_sky.sum())}, "
                    f"grad_thresh={grad_thresh:.3f}"
                )

    final_sky = sure_sky | expanded_sky
    final_mask_obj = ~final_sky

    # 5. winner-takes-depth：
    # 最终标签与哪一方一致，就采用哪一方的深度。
    follow_da3 = (final_mask_obj == is_da3_obj)
    final_depth = dggt_depth_np.copy()
    final_depth[follow_da3] = da3_aligned[follow_da3]

    # 6. 生成 Final Mask (1=Sky, 0=Object)
    final_sky_mask = final_sky.astype(np.uint8) # 1=Sky
    
    return final_depth, final_sky_mask, da3_aligned


# -----------------------------------------------------------------------------
# Full Sequence Inference Utilities
# -----------------------------------------------------------------------------

class FileLogger:
    """
    Logger that outputs to both console and file.
    Each scene can have its own log file.
    """
    def __init__(self, log_dir, scene_name, console_output=True):
        """
        Args:
            log_dir: Directory to save log files
            scene_name: Scene identifier for the log file name
            console_output: Whether to also print to console
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"scene_{scene_name}.log")
        self.console_output = console_output
        
        # Create/clear the log file
        with open(self.log_path, 'w') as f:
            f.write(f"=== Log for Scene {scene_name} ===\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def info(self, msg):
        """Log an info message."""
        timestamp = time.strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] [INFO] {msg}"
        
        # Write to file
        with open(self.log_path, 'a') as f:
            f.write(formatted_msg + "\n")
        
        # Print to console
        if self.console_output:
            print(formatted_msg)
    
    def warning(self, msg):
        """Log a warning message."""
        timestamp = time.strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] [WARN] {msg}"
        
        with open(self.log_path, 'a') as f:
            f.write(formatted_msg + "\n")
        
        if self.console_output:
            print(formatted_msg)
    
    def error(self, msg):
        """Log an error message."""
        timestamp = time.strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] [ERROR] {msg}"
        
        with open(self.log_path, 'a') as f:
            f.write(formatted_msg + "\n")
        
        if self.console_output:
            print(formatted_msg)
    
    def close(self):
        """Finalize the log file."""
        with open(self.log_path, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def count_frames_for_scene(image_dir, scene_name):
    """
    统计场景的帧数（以 _0.jpg 结尾的图片数量）
    
    Args:
        image_dir: 数据根目录
        scene_name: 场景名称
    
    Returns:
        帧数
    """
    pattern = os.path.join(image_dir, scene_name, "images_4", "*_0.jpg")
    files = glob.glob(pattern)
    return len(files)


def run_full_sequence_inference(
    config, model, dataset_class, scene_name, device='cuda',
    sequence_length=20, overlap=3, logger=None
):
    """
    对完整序列进行推理，使用 overlap 策略处理长序列。
    
    Args:
        config: DGGTConfig 实例
        model: 加载好的模型
        dataset_class: Dataset 类（如 WaymoOpenDataset）
        scene_name: 场景名称
        device: 设备
        sequence_length: 每个窗口的帧数
        overlap: 窗口间的重叠帧数
        logger: 日志记录器
    
    Returns:
        dict: {
            'rendered_images': [N, C, H, W] 完整序列的渲染结果,
            'gt_images': [N, C, H, W] 完整序列的 GT 图像
        }
    """
    # 统计总帧数
    total_frames = count_frames_for_scene(config.image_dir, scene_name)
    if logger:
        logger.info(f"Scene {scene_name}: Total {total_frames} frames")
    
    if total_frames == 0:
        if logger:
            logger.error(f"No frames found for scene {scene_name}")
        return None
    
    all_rendered_images = []
    all_gt_images = []
    
    # 计算窗口参数
    step = sequence_length - overlap
    num_windows = (total_frames - overlap + step - 1) // step  # 向上取整
    
    if logger:
        logger.info(f"Window strategy: length={sequence_length}, overlap={overlap}, step={step}")
        logger.info(f"Estimated windows: {num_windows}")
    
    window_idx = 0
    current_start = 0
    
    while current_start < total_frames:
        # 计算当前窗口的帧范围
        window_end = min(current_start + sequence_length, total_frames)
        actual_length = window_end - current_start
        
        if logger:
            logger.info(f"Window {window_idx}: frames [{current_start}, {window_end}) ({actual_length} frames)")
        
        # 创建临时 config - 使用 replace() 复制所有参数，只修改需要变化的字段
        temp_config = replace(
            config,
            scene_names=[scene_name],
            sequence_length=actual_length,
            start_idx=current_start,
            metrics=False  # 单独窗口不计算metrics，最后统一计算
        )
        
        # 创建数据集
        temp_dataset = dataset_class(
            image_dir=config.image_dir,
            scene_names=[scene_name],
            sequence_length=actual_length,
            start_idx=current_start,
            mode=2,
            views=config.input_views
        )
        
        # 运行推理
        results = run_inference_and_render(temp_config, model, temp_dataset, logger, device=device)
        
        if results and len(results) > 0:
            rendered_images = results[0]['rendered_image']  # [T, C, H, W]
            gt_images = results[0]['gt_image']  # [T, C, H, W]
            
            # 决定保留哪些帧
            if window_idx == 0:
                # 第一个窗口：保留所有帧
                frames_to_keep = rendered_images
                gt_to_keep = gt_images
            else:
                # 后续窗口：只保留后 (actual_length - overlap) 帧
                discard_count = min(overlap, actual_length - 1)
                frames_to_keep = rendered_images[discard_count:]
                gt_to_keep = gt_images[discard_count:]
            
            if logger:
                logger.info(f"  Window {window_idx}: keeping {frames_to_keep.shape[0]} frames")
            
            all_rendered_images.append(frames_to_keep)
            all_gt_images.append(gt_to_keep)
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 移动到下一个窗口
        current_start += step
        window_idx += 1
        
        # 如果下一个窗口起始位置已经超过或等于总帧数，退出
        if current_start >= total_frames:
            break
            
        # 如果剩余帧数太少（小于 overlap），合并到最后一个窗口处理
        remaining = total_frames - current_start
        if remaining <= overlap and window_idx > 0:
            if logger:
                logger.info(f"  Remaining {remaining} frames <= overlap, stopping")
            break
    
    # 合并所有帧
    if all_rendered_images:
        all_rendered_images = torch.cat(all_rendered_images, dim=0)
        all_gt_images = torch.cat(all_gt_images, dim=0)
        if logger:
            logger.info(f"Total rendered frames: {all_rendered_images.shape[0]}")
        return {
            'rendered_images': all_rendered_images,
            'gt_images': all_gt_images
        }
    
    return None


def process_single_scene(
    config, model, dataset_class, scene_name, device='cuda',
    sequence_length=20, overlap=3, output_dir="outputs/full_sequence",
    save_video_func=None, fps=8, logger=None
):
    """
    处理单个场景的完整序列推理。
    
    Args:
        config: DGGTConfig 实例
        model: 加载好的模型
        dataset_class: Dataset 类
        scene_name: 场景名称
        device: 设备
        sequence_length: 每个窗口的帧数
        overlap: 窗口间的重叠帧数
        output_dir: 输出目录
        save_video_func: 保存视频的函数
        fps: 视频帧率
        logger: 日志记录器（可选）
    
    Returns:
        result: {'video_path': str, 'metrics': dict, 'num_frames': int}
    """
    # 初始化 LPIPS (如果需要计算 metrics)
    loss_fn = None
    if config.metrics:
        loss_fn = lpips.LPIPS(net='alex').to(device)
    
    scene_result = {'video_path': None, 'metrics': None, 'num_frames': 0}
    
    try:
        # 运行完整序列推理
        inference_result = run_full_sequence_inference(
            config=config,
            model=model,
            dataset_class=dataset_class,
            scene_name=scene_name,
            device=device,
            sequence_length=sequence_length,
            overlap=overlap,
            logger=logger
        )
        
        if inference_result is not None:
            all_rendered = inference_result['rendered_images']
            all_gt = inference_result['gt_images']
            scene_result['num_frames'] = all_rendered.shape[0]
            
            # 计算完整序列的 metrics
            if config.metrics and loss_fn is not None:
                if logger:
                    logger.info("Computing metrics for full sequence...")
                
                # 将数据移到正确的设备
                all_rendered_device = all_rendered.to(device)
                all_gt_device = all_gt.to(device)
                
                psnr, ssim, lpip = compute_metrics(all_rendered_device, all_gt_device, loss_fn)
                
                scene_result['metrics'] = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpip
                }
                
                if logger:
                    logger.info(f"Scene {scene_name} Full Sequence Metrics:")
                    logger.info(f"  PSNR:   {psnr:.4f}")
                    logger.info(f"  SSIM:   {ssim:.4f}")
                    logger.info(f"  LPIPS:  {lpip:.4f}")
                    logger.info(f"  Frames: {all_rendered.shape[0]}")
            
            # 保存视频和帧图片
            if save_video_func is not None:
                video_path = save_video_func(
                    images=all_rendered,
                    output_dir=output_dir,
                    scene_name=scene_name,
                    logger=logger,
                    fps=fps
                )
                if logger:
                    logger.info(f"Scene {scene_name} complete! Video saved to: {video_path}")
                scene_result['video_path'] = video_path
        else:
            if logger:
                logger.error(f"Scene {scene_name} failed or no frames found.")
            
    except Exception as e:
        if logger:
            logger.error(f"Exception during scene {scene_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return scene_result