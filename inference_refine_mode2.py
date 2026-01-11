"""
inference_refine_mode2.py - Simplified version for mode=2 only with Test-Time Refinement
"""
import argparse
import os
import time
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.losses import ssim_loss
from third_party.difix.infer import process_images_with_difix
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset
from utils.video_maker import make_comparison_video_quad


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


# ========== Rendering Helper Functions ==========

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


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    """Temporal opacity modulation."""
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()


def render_all_frames(static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                      extrinsic, intrinsic, H, W, sky_model, images):
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
    renders = alphas * renders + (1 - alphas) * bg_render
    
    rendered_images = renders.permute(0, 3, 1, 2)
    return rendered_images, depth_maps, alphas


# ========== Logger & Utilities ==========

def setup_logger(output_path):
    log_file = os.path.join(output_path, 'refinement.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_side_by_side_video(gt_images, pred_images, output_path, fps=8):
    """Saves a side-by-side comparison video."""
    frames = []
    num_frames = min(len(gt_images), len(pred_images))
    
    gt_images = gt_images.detach().cpu()
    pred_images = pred_images.detach().cpu()
    
    for i in range(num_frames):
        gt_frame = gt_images[i]
        pred_frame = pred_images[i]
        
        if gt_frame.shape[0] == 3:
            gt_frame = gt_frame.permute(1, 2, 0)
        if pred_frame.shape[0] == 3:
            pred_frame = pred_frame.permute(1, 2, 0)
            
        gt_np = (gt_frame.clamp(0, 1).numpy() * 255).astype(np.uint8)
        pred_np = (pred_frame.clamp(0, 1).numpy() * 255).astype(np.uint8)
        
        H, W, C = gt_np.shape
        spacer = np.ones((H, 10, C), dtype=np.uint8) * 255
        
        combined_frame = np.concatenate([gt_np, spacer, pred_np], axis=1)
        frames.append(combined_frame)
        
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264')
    logging.info(f"Saved side-by-side video to {output_path}")


def compute_metrics(img1, img2, loss_fn):
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


def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end)+1)]
    else:
        return [str(int(x)).zfill(3) for x in scene_names_str.split()]


def apply_diffusion(images, model_path, device):
    """Apply diffusion model to a batch of images."""
    processed_frames = []
    for i in range(images.shape[0]):
        frame = images[i].detach().cpu().clamp(0, 1)
        processed_frame = process_images_with_difix(frame, model_path)
        processed_frames.append(processed_frame)
    return torch.stack(processed_frames, dim=0).to(device)


# ========== Cache Utilities ==========

def get_baseline_cache_dir(scene_name, start_idx, sequence_length):
    """Get the baseline cache directory path."""
    cache_dir = f"./outputs/baseline/{scene_name}/{start_idx}_{sequence_length}"
    return cache_dir


def save_predictions_cache(cache_dir, predictions, logger):
    """Save model predictions to cache using torch.save."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "predictions.pt")
    
    # Convert predictions to CPU tensors for storage
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


def save_diffusion_cache(cache_dir, rendered_image, cache_name, logger):
    """Save diffusion-processed images to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"diffusion_{cache_name}.pt")
    torch.save(rendered_image.detach().cpu(), cache_path)
    logger.info(f"Saved diffusion cache ({cache_name}) to {cache_path}")


def load_diffusion_cache(cache_dir, cache_name, device, logger):
    """Load diffusion-processed images from cache."""
    cache_path = os.path.join(cache_dir, f"diffusion_{cache_name}.pt")
    if not os.path.exists(cache_path):
        return None
    
    cached_image = torch.load(cache_path, map_location=device)
    logger.info(f"Loaded diffusion cache ({cache_name}) from {cache_path}")
    return cached_image


def save_images_and_video(images, output_dir, input_views, logger, name="rendered"):
    """
    Save rendered images and video to specified directory.
    
    Args:
        images: [T, C, H, W] tensor, values in [0, 1]
        output_dir: directory to save images and video
        input_views: number of input views (1 or 3)
        logger: logger instance
        name: name for logging
    """
    os.makedirs(output_dir, exist_ok=True)
    images_cpu = images.detach().cpu()
    
    def to_uint8(arr):
        """Convert numpy array to uint8."""
        a = (arr * 255.0).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.shape[2] == 4:
            a = a[:, :, :3]
        return a
    
    if input_views == 1:
        image_list = []
        for i in range(images_cpu.shape[0]):
            rendered = images_cpu[i].clamp(0, 1)
            image_path = os.path.join(output_dir, f"view_{i}.png")
            T.ToPILImage()(rendered).save(image_path)
            image_list.append(rendered.permute(1, 2, 0).numpy())
        video_path = os.path.join(output_dir, "rendered_video.mp4")
        imageio.mimwrite(video_path, (np.array(image_list) * 255).astype(np.uint8), fps=8, codec="libx264")
        logger.info(f"Saved {name} images ({images_cpu.shape[0]} frames) and video to {output_dir}")
        
    elif input_views == 3:
        T_total = images_cpu.shape[0]
        groups = T_total // 3
        video_list = []
        for g in range(groups):
            idx_center = 3 * g + 0
            idx_left = 3 * g + 1
            idx_right = 3 * g + 2
            center = images_cpu[idx_center].clamp(0, 1).permute(1, 2, 0).numpy()
            left = images_cpu[idx_left].clamp(0, 1).permute(1, 2, 0).numpy()
            right = images_cpu[idx_right].clamp(0, 1).permute(1, 2, 0).numpy()
            H_img, W_img = center.shape[0], center.shape[1]
            
            left_u = to_uint8(left)
            center_u = to_uint8(center)
            right_u = to_uint8(right)
            white = np.ones((H_img, 10, 3), dtype=np.uint8) * 255
            composed = np.concatenate([left_u, white, center_u, white, right_u], axis=1)
            
            image_path = os.path.join(output_dir, f"view_{g:04d}.png")
            Image.fromarray(composed).save(image_path)
            video_list.append(composed)
        
        video_path = os.path.join(output_dir, "rendered_video.mp4")
        imageio.mimwrite(video_path, np.array(video_list), fps=8, codec="libx264")
        logger.info(f"Saved {name} images ({groups} frames) and video to {output_dir}")


# ========== Refinement Trainer ==========

class RefinementTrainer:
    def __init__(self, args, logger, device='cuda'):
        self.args = args
        self.logger = logger
        self.device = device
        self.enable_refinement = args.enable_refinement
        self.refine_pose = args.refine_pose
        self.refine_gs = args.refine_gs
        self.refine_xyz = args.refine_xyz
        self.num_steps_pose = args.refine_steps_pose if args.refine_steps_pose is not None else args.refine_steps
        self.num_steps_gs = args.refine_steps_gs if args.refine_steps_gs is not None else args.refine_steps
        self.lr_pose = args.refine_lr_pose
        self.lr_opacity = args.refine_lr_opacity
        self.lr_color = args.refine_lr_color
        self.lr_rot = args.refine_lr_rot
        self.lr_scales = args.refine_lr_scales
        self.lr_xyz = args.refine_lr_xyz
        self.pose_lr_factor_gs = args.pose_lr_factor_gs
        
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
        # total_loss = l1_loss
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
            # noise_mask = (alphas < 0.2).float().detach()
            # penalty = (alphas * noise_mask).sum() / (noise_mask.sum() + 1e-6)
            # total_loss = total_loss + penalty
            # # Sky opacity loss: 惩罚天空区域的不透明高斯球
            # sky_opacity_loss = torch.tensor(0.0, device=self.device)
            # if loss_mask is not None:
            #     # loss_mask: [T, H, W], 1 = valid (非天空), 0 = sky
            #     # sky_mask: [T, H, W], 1 = sky, 0 = valid
            #     sky_mask = 1.0 - loss_mask  # [T, H, W]
            #     sky_pixels = sky_mask.sum() + 1e-8
                
            #     # alphas: [T, H, W, 1] -> [T, H, W]
            #     alphas_2d = alphas.squeeze(-1)  # [T, H, W]
                
            #     # Skip first frame if needed
            #     if alphas_2d.shape[0] > 1:
            #         alphas_2d = alphas_2d[1:]
            #         sky_mask = sky_mask[1:]
            #         sky_pixels = sky_mask.sum() + 1e-8
                
            #     # Penalize opacity in sky regions
            #     sky_opacity_loss = (alphas_2d * sky_mask).sum() / sky_pixels
            # total_loss = total_loss + 0.03 * sky_opacity_loss
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_pose_delta = pose_delta.clone().detach()
                best_static = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in current_static_gs.items()}
            
            if step % 10 == 0 or step == num_steps - 1:
                # sky_loss_str = f", SkyOpacity={sky_opacity_loss.item():.6f}" if loss_mask is not None else ""
                sky_loss_str = ""
                self.logger.info(f"  Step {step:3d}/{num_steps}: Loss={total_loss.item():.6f} (L1={l1_loss.item():.6f}, SSIM={ssim_val.item():.6f}{sky_loss_str})")
        
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


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Inference with Test-Time Refinement (Mode=2 Only)")
    
    # Basic arguments
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the input images')
    parser.add_argument('--scene_names', type=str, nargs='+', required=True, help='Scene names')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    parser.add_argument('-images', action='store_true', help='Whether to output each frame image')
    parser.add_argument('-depth', action='store_true', help='Whether to output each frame depth as .npy')
    parser.add_argument('-metrics', action='store_true', help='Whether to output evaluation metrics')
    parser.add_argument('-diffusion', action='store_true', help='Whether to process images with diffusion model')

    # Refinement arguments
    parser.add_argument('--enable_refinement', action='store_true', help='Enable test-time refinement')
    parser.add_argument('--refine_pose', action='store_true', help='Enable pose optimization')
    parser.add_argument('--refine_gs', action='store_true', help='Enable GS outputs optimization (opacity/color/rot)')
    parser.add_argument('--refine_xyz', action='store_true', help='Enable GS xyz optimization (only effective when refine_gs is True)')
    parser.add_argument('--refine_steps', type=int, default=50, help='Total refinement steps (legacy)')
    parser.add_argument('--refine_steps_pose', type=int, default=50, help='Pose-only refinement steps')
    parser.add_argument('--refine_steps_gs', type=int, default=50, help='GS refinement steps')
    parser.add_argument('--pose_lr_factor_gs', type=float, default=0.1, help='Scale pose LR during GS stage (0 freezes pose)')
    parser.add_argument('--refine_lr_pose', type=float, default=1e-4, help='Learning rate for pose optimization')
    parser.add_argument('--refine_lr_opacity', type=float, default=0.05, help='Learning rate for opacity optimization')
    parser.add_argument('--refine_lr_color', type=float, default=0.003, help='Learning rate for color/SH optimization')
    parser.add_argument('--refine_lr_rot', type=float, default=0.001, help='Learning rate for rotation optimization')
    parser.add_argument('--refine_lr_scales', type=float, default=1e-5, help='Learning rate for scales optimization')
    parser.add_argument('--refine_lr_xyz', type=float, default=1e-4, help='Learning rate for xyz optimization (very small)')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_path)
    logger.info(f"Output directory: {args.output_path}")
    logger.info(f"Arguments: {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    loss_fn = lpips.LPIPS(net='alex').to(device)

    scene_names_str = ' '.join(args.scene_names)
    scene_names = parse_scene_names(scene_names_str)
    
    # Mode=2 only dataset
    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=scene_names,
        sequence_length=args.sequence_length,
        start_idx=args.start_idx,
        mode=2,
        views=args.input_views
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    psnr_list, ssim_list, lpips_list = [], [], []
    inference_time_list = []
    scene_idx = 1
    
    # Initialize Refinement Trainer
    refiner = RefinementTrainer(args, logger)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            nearest_sky_mask = batch['nearest_masks'].to(device).permute(0, 1, 3, 4, 2)
            gt_dy_map = batch['dynamic_mask'].to(device)
            gt_depth = batch['gt_depth'].to(device)

            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            start_time = time.time()

            # Get scene name for cache path
            scene_name = scene_names[scene_idx - 1] if scene_idx <= len(scene_names) else str(scene_idx).zfill(3)
            cache_dir = get_baseline_cache_dir(scene_name, args.start_idx, args.sequence_length)

            # --- DGGT Feedforward Inference (with caching) ---
            predictions = load_predictions_cache(cache_dir, device, logger)
            
            if predictions is None:
                logger.info("No predictions cache found, running model inference...")
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
                    # Save to cache
                    save_predictions_cache(cache_dir, predictions, logger)
            else:
                logger.info("Using cached predictions (skipping model inference)")
            
            with torch.cuda.amp.autocast(dtype=dtype):
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                use_depth = True
                if use_depth:
                    depth_map = predictions["depth"][0]
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:
                    point_map = predictions["world_points"]
                    
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1)

                # Mode=2 processing
                static_mask = (bg_mask & (dy_map < 0.5))
                static_points = point_map[static_mask].reshape(-1, 3)
                gs_dynamic_list = dy_map[static_mask].sigmoid()
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                static_opacity = static_opacity * (1 - gs_dynamic_list)
                static_gs_conf = gs_conf[static_mask]
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                gs_timestamps = timestamps[frame_idx]

                # Pack static Gaussians into dict
                static_gs = {
                    'points': static_points,
                    'rgbs': static_rgbs,
                    'opacity': static_opacity,
                    'scales': static_scales,
                    'rotations': static_rotations,
                    'gs_conf': static_gs_conf
                }
                
                # Prepare dynamic Gaussians list
                dynamic_gs_list = []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
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
                
                # === Step 1: Render Initial (before refinement) ===
                logger.info("Rendering Initial (before refinement)...")
                rendered_image_init, depth_maps, alphas = render_all_frames(
                    static_gs, dynamic_gs_list, timestamps, gs_timestamps,
                    extrinsic, intrinsic, H, W, model.sky_model, images
                )

        # Refinement needs gradients, so we do it outside the no_grad block
        optimized_extrinsic = extrinsic
        optimized_static_gs = static_gs

        if args.enable_refinement:
            logger.info("Starting Test-Time Refinement...")
            # Construct nearest_static_gs using nearest_masks for refinement
            logger.info("Constructing nearest_static_gs using nearest_masks...")
            bg_mask_nearest = (nearest_sky_mask == 0).any(dim=-1)
            static_mask_nearest = (bg_mask_nearest & (dy_map < 0.5))
            static_points_nearest = point_map[static_mask_nearest].reshape(-1, 3)
            gs_dynamic_list_nearest = dy_map[static_mask_nearest].sigmoid()
            static_rgbs_nearest, static_opacity_nearest, static_scales_nearest, static_rotations_nearest = get_split_gs(gs_map, static_mask_nearest)
            static_opacity_nearest = static_opacity_nearest * (1 - gs_dynamic_list_nearest)
            static_gs_conf_nearest = gs_conf[static_mask_nearest]
            frame_idx_nearest = torch.nonzero(static_mask_nearest, as_tuple=False)[:,1]
            gs_timestamps_nearest = timestamps[frame_idx_nearest]
            
            nearest_static_gs = {
                'points': static_points_nearest,
                'rgbs': static_rgbs_nearest,
                'opacity': static_opacity_nearest,
                'scales': static_scales_nearest,
                'rotations': static_rotations_nearest,
                'gs_conf': static_gs_conf_nearest
            }
            
            # Prepare dynamic Gaussians list using nearest_masks
            dynamic_gs_list_nearest = []
            for i in range(dy_map.shape[1]):
                point_map_i = point_map[:, i]
                bg_mask_i_nearest = bg_mask_nearest[:, i]
                dynamic_point_nearest = point_map_i[bg_mask_i_nearest].reshape(-1, 3)
                dynamic_rgb_nearest, dynamic_opacity_nearest, dynamic_scale_nearest, dynamic_rotation_nearest = get_split_gs(gs_map[:, i], bg_mask_i_nearest)
                gs_dynamic_list_i_nearest = dy_map[:, i][bg_mask_i_nearest].sigmoid()
                dynamic_opacity_nearest = dynamic_opacity_nearest * gs_dynamic_list_i_nearest
                
                dynamic_gs_list_nearest.append({
                    'points': dynamic_point_nearest,
                    'rgbs': dynamic_rgb_nearest,
                    'opacity': dynamic_opacity_nearest,
                    'scales': dynamic_scale_nearest,
                    'rotations': dynamic_rotation_nearest
                })
            
            with torch.enable_grad():
                optimized_extrinsic, optimized_static_gs = refiner.forward(
                    gt_images=target_image,
                    static_gs=nearest_static_gs,
                    dynamic_gs_list=dynamic_gs_list_nearest,
                    timestamps=timestamps,
                    gs_timestamps=gs_timestamps_nearest,
                    extrinsic=extrinsic,
                    intrinsic=intrinsic,
                    H=H, W=W,
                    sky_model=model.sky_model,
                    images=images,
                    sky_mask=nearest_sky_mask[0] if nearest_sky_mask is not None else None,
                    dynamic_mask=gt_dy_map[0] if gt_dy_map is not None else None
                )
            
            # === Step 3: Render Final (after refinement) ===
            logger.info("Rendering Final (after refinement)...")
            with torch.no_grad():
                rendered_image, depth_maps_refined, alphas_refined = render_all_frames(
                    optimized_static_gs, dynamic_gs_list, timestamps, gs_timestamps_nearest,
                    optimized_extrinsic, intrinsic, H, W, model.sky_model, images
                )
        else:
            # No refinement, use initial render
            rendered_image = rendered_image_init

        scene_out_name = str(scene_idx).zfill(3)
        inference_time = time.time() - start_time
        inference_time_list.append(inference_time)
        
        # Create output directory
        scene_out_dir = os.path.join(args.output_path, scene_out_name)
        os.makedirs(scene_out_dir, exist_ok=True)
        
        # === Apply Diffusion (if enabled) ===
        # rendered_image_final: 最终结果（可能经过 refinement + diffusion）
        # rendered_image_init_final: 初始结果（可能经过 diffusion，用于 baseline 对比）
        
        if args.enable_refinement:
            # 有 refinement: final 来自 refined 结果，init 来自初始结果
            rendered_image_final = rendered_image.clone()
            rendered_image_init_final = rendered_image_init.clone()
        else:
            # 无 refinement: final 和 init 相同
            rendered_image_final = rendered_image_init.clone()
            rendered_image_init_final = None  # 不需要单独的 baseline
        
        if args.diffusion:
            rendered_image_final = apply_diffusion(rendered_image_final, "/root/autodl-tmp/dggt/pretrained/diffusion_model.pth", device)
            
            # 对 init 结果应用 diffusion（仅当有 refinement 时才需要单独处理）
            if args.enable_refinement:
                cached_diffusion_init = load_diffusion_cache(cache_dir, "init", device, logger)
                if cached_diffusion_init is not None:
                    logger.info("Using cached diffusion result for Initial render")
                    rendered_image_init_final = cached_diffusion_init
                else:
                    logger.info("Applying diffusion to Initial render...")
                    rendered_image_init_final = apply_diffusion(rendered_image_init_final, "/root/autodl-tmp/dggt/pretrained/diffusion_model.pth", device)
                    save_diffusion_cache(cache_dir, rendered_image_init_final, "init", logger)
        
        # === Compute Metrics ===
        psnr, ssim, lpip = compute_metrics(rendered_image_final, target_image, loss_fn)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpip)
        
        # Log comparison metrics
        if args.enable_refinement and rendered_image_init_final is not None:
            psnr_init, ssim_init, lpip_init = compute_metrics(rendered_image_init_final, target_image, loss_fn)
            logger.info(f"Initial Metrics: PSNR={psnr_init:.4f}, SSIM={ssim_init:.4f}, LPIPS={lpip_init:.4f}")
            logger.info(f"Final (Refined) Metrics: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpip:.4f}")
            psnr_diff = psnr - psnr_init
            ssim_diff = ssim - ssim_init
            lpip_diff = lpip - lpip_init
            logger.info(f"Improvement: PSNR {'+' if psnr_diff >= 0 else ''}{psnr_diff:.4f}, SSIM {'+' if ssim_diff >= 0 else ''}{ssim_diff:.4f}, LPIPS {lpip_diff:.4f}")
        
        scene_idx += 1

        # === Save Images and Videos ===
        # 如果有 refinement，保存两份（baseline 和 final）
        # 如果没有 refinement，只保存一份（final）
        
        if args.enable_refinement and rendered_image_init_final is not None:
            # 保存 Baseline（初始结果，可能经过 diffusion）
            baseline_out_dir = os.path.join(scene_out_dir, "baseline")
            save_images_and_video(rendered_image_init_final, baseline_out_dir, args.input_views, logger, name="baseline")
        
        # 保存 Final 结果（可能经过 refinement + diffusion）
        if args.images:
            save_images_and_video(rendered_image_final, scene_out_dir, args.input_views, logger, name="final")

        # Save comparison videos
        gt_frames = target_image.detach().cpu()
        pred_frames = rendered_image_final.detach().cpu()
        dyn_frames = dy_map[0].sigmoid().detach().cpu()
        gt_dy_map_cpu = gt_dy_map.mean(dim=2)[0].sigmoid().detach().cpu()
        
        depth_frames = predictions["depth"][0].detach().cpu()
        gt_depth_cpu = gt_depth[..., 0:1][0].squeeze(-1).detach().cpu()
        sky_mask_cpu = sky_mask.detach().cpu()
        
        out_video = os.path.join(scene_out_dir, "comparison.mp4")
        make_comparison_video_quad(gt_frames, pred_frames, gt_dy_map_cpu, dyn_frames, gt_depth_cpu, depth_frames, sky_mask_cpu, out_video, fps=8, views=args.input_views)
        logger.info(f"Saved comparison video: {out_video}")

        # Save Side-by-Side Video (GT vs Pred)
        sbs_video_path = os.path.join(scene_out_dir, "side_by_side.mp4")
        save_side_by_side_video(gt_frames, pred_frames, sbs_video_path, fps=8)
        
        # Save Initial vs Refined Video
        if args.enable_refinement and rendered_image_init_final is not None:
            init_frames = rendered_image_init_final.detach().cpu()
            init_vs_refined_path = os.path.join(scene_out_dir, "init_vs_refined.mp4")
            save_side_by_side_video(init_frames, pred_frames, init_vs_refined_path, fps=8)
            logger.info(f"Saved Init vs Refined video: {init_vs_refined_path}")

        # Save depth
        if args.depth:
            S = depth_frames.shape[0]
            if args.input_views == 1:
                for i in range(S):
                    depth_i = depth_frames[i].numpy()
                    np.save(os.path.join(scene_out_dir, f"view_{i}.npy"), depth_i)
            elif args.input_views == 3:
                for i in range(S):
                    view_id = i % 3
                    frame_id = i // 3
                    depth_i = depth_frames[i].numpy()
                    np.save(os.path.join(scene_out_dir, f"view_{frame_id:04d}_{view_id}.npy"), depth_i)

    # Final metrics
    if args.metrics:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_lpips = sum(lpips_list) / len(lpips_list)
        avg_inference_time = sum(inference_time_list) / len(inference_time_list)
        
        logger.info(f"Final Metrics for Scene {scene_names}:")
        logger.info(f"PSNR: {avg_psnr:.4f}")
        logger.info(f"SSIM: {avg_ssim:.4f}")
        logger.info(f"LPIPS: {avg_lpips:.4f}")
        logger.info(f"Avg Inference Time (s): {avg_inference_time:.4f}")
        
        print("PSNR:", avg_psnr)
        print("SSIM:", avg_ssim)
        print("LPIPS:", avg_lpips)
        print("Avg Inference Time (s):", avg_inference_time)


if __name__ == "__main__":
    main()
