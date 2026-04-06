import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import imageio
import torchvision.transforms as T
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Visualization Helper Functions
# -----------------------------------------------------------------------------


def tensor_to_img_numpy(tensor):
    """
    Convert tensor [C, H, W] (normalized or 0-1) to numpy [H, W, C] (0-1).
    """
    img = tensor.permute(1, 2, 0).cpu().numpy()
    # Handle simple normalization if necessary, usually assuming input is 0-1
    return np.clip(img, 0, 1)

def apply_overlay(image, mask, color=(1, 0, 0), alpha=0.4):
    """
    image: [H, W, 3] numpy, 0-1
    mask: [H, W, C] or [H, W] numpy. 
          If mask is soft (0-1), it acts as opacity.
          If mask is binary, it acts as strict overlay.
    color: tuple (R, G, B), default Red.
    """
    if mask.ndim == 3:
        # If mask is 3 channels, take the first one or average, or use logic
        # Typically sky mask might be 3 channel black/white.
        # User logic: bg_mask = (sky_mask == 0).any(dim=-1)
        # Let's visualize the "Sky" part.
        # Assuming Mask is: 0=Sky, 1=Object (or vice versa based on user description)
        # Let's treat the mask input as the "Attention Area" we want to highlight.
        mask_layer = mask.mean(axis=-1) if mask.shape[-1] == 3 else mask.squeeze()
    else:
        mask_layer = mask

    # Create colored overlay
    overlay = image.copy()
    colored_layer = np.zeros_like(image)
    for c in range(3):
        colored_layer[:, :, c] = color[c]
        
    # Expand mask for broadcasting
    mask_layer = mask_layer[:, :, None]
    
    # Blend: Original * (1 - alpha*mask) + Color * (alpha*mask)
    # Highlight regions where mask > 0.1
    # Note: For Bicubic, mask values will be gradients 0..1 at edges.
    output = image * (1 - alpha * mask_layer) + colored_layer * (alpha * mask_layer)
    return np.clip(output, 0, 1)

def _compose_triview(center, left, right, spacer_width=10):
    """Horizontally concatenate left | center | right with white spacers.
    All inputs are [H, W, 3] numpy arrays in 0-1 range.
    """
    H = center.shape[0]
    spacer = np.ones((H, spacer_width, 3))
    return np.concatenate([left, spacer, center, spacer, right], axis=1)


def compute_color_gradient_map(image_np):
    """
    Compute normalized color gradient magnitude map from RGB image [H, W, 3] in [0, 1].
    Returns [H, W] in [0, 1].
    """
    img = np.clip(image_np, 0, 1).astype(np.float32)
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Robust normalization to keep visualization stable across scenes
    high = np.percentile(grad_mag, 99)
    if high > 1e-8:
        grad_mag = grad_mag / high
    return np.clip(grad_mag, 0, 1)


def visualize_samples(dataset, sample_indices, output_path=None, views=1):
    """
    Extracts one batch (sequence) and visualizes specific frames.
    When views=3, sample_indices refer to *group* indices (each group = 3 interleaved frames).
    """
    data_dict = dataset[0] 
    
    images = data_dict['images']         # [S, C, H, W]  (S = seq*views)
    masks_bicubic = data_dict['masks']   # [S, C, H, W]
    masks_nearest = data_dict['nearest_masks'] # [S, C, H, W]
    
    has_dynamic = 'dynamic_mask' in data_dict
    if has_dynamic:
        masks_dynamic = data_dict['dynamic_mask'] # [S, C, H, W]

    total_frames = images.shape[0]
    num_groups = total_frames // views if views == 3 else total_frames
    print(f"Loaded sequence: {total_frames} frames, views={views}, groups={num_groups}")
    print(f"Image shape: {images.shape}")
    
    for group_idx in sample_indices:
        if group_idx >= num_groups:
            print(f"Skipping index {group_idx}, out of bounds (groups={num_groups})")
            continue

        if views == 3:
            idxs = [3 * group_idx + v for v in range(3)]  # center, left, right
            print(f"Visualizing Group {group_idx} (frames {idxs})...")
        else:
            idxs = [group_idx]
            print(f"Visualizing Frame {group_idx}...")

        imgs_np = [tensor_to_img_numpy(images[i]) for i in idxs]
        masks_bi_np = [tensor_to_img_numpy(masks_bicubic[i]) for i in idxs]
        masks_near_np = [tensor_to_img_numpy(masks_nearest[i]) for i in idxs]

        sky_bi = [(m == 0).any(axis=-1).astype(float) for m in masks_bi_np]
        sky_near = [(m == 0).any(axis=-1).astype(float) for m in masks_near_np]
        grad_maps = [compute_color_gradient_map(img) for img in imgs_np]
        grad_overlay = [apply_overlay(imgs_np[v], grad_maps[v], color=(1, 1, 0), alpha=0.6) for v in range(len(idxs))]

        if has_dynamic:
            masks_dyn_np = [tensor_to_img_numpy(masks_dynamic[i]) for i in idxs]
            is_dyn = [(m > 0.5).any(axis=-1).astype(float) for m in masks_dyn_np]

        if views == 3:
            img_comp = _compose_triview(imgs_np[0], imgs_np[1], imgs_np[2])
            ov_bi_list = [apply_overlay(imgs_np[v], sky_bi[v], color=(1, 0, 0), alpha=0.5) for v in range(3)]
            ov_bi_comp = _compose_triview(ov_bi_list[0], ov_bi_list[1], ov_bi_list[2])
            ov_near_list = [apply_overlay(imgs_np[v], sky_near[v], color=(0, 1, 0), alpha=0.5) for v in range(3)]
            ov_near_comp = _compose_triview(ov_near_list[0], ov_near_list[1], ov_near_list[2])
            grad_overlay_comp = _compose_triview(grad_overlay[0], grad_overlay[1], grad_overlay[2])

            rows = 5 if has_dynamic else 4
            fig, axes = plt.subplots(rows, 1, figsize=(20, 5 * rows))
            plt.suptitle(f"Group {group_idx} (Left | Center | Right)", fontsize=16)

            axes[0].imshow(img_comp); axes[0].set_title("Original"); axes[0].axis('off')
            axes[1].imshow(ov_bi_comp); axes[1].set_title("Bicubic Sky (Red)"); axes[1].axis('off')
            axes[2].imshow(ov_near_comp); axes[2].set_title("Nearest Sky (Green)"); axes[2].axis('off')
            axes[3].imshow(grad_overlay_comp); axes[3].set_title("Color Gradient Overlay (Yellow)"); axes[3].axis('off')

            if has_dynamic:
                ov_dyn_list = [apply_overlay(imgs_np[v], is_dyn[v], color=(0, 0, 1), alpha=0.6) for v in range(3)]
                ov_dyn_comp = _compose_triview(ov_dyn_list[0], ov_dyn_list[1], ov_dyn_list[2])
                axes[4].imshow(ov_dyn_comp); axes[4].set_title("Dynamic (Blue)"); axes[4].axis('off')
        else:
            img_np = imgs_np[0]
            cols = 5 if has_dynamic else 4
            fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
            plt.suptitle(f"Frame {group_idx} Analysis", fontsize=16)

            axes[0].imshow(img_np); axes[0].set_title("Original Image"); axes[0].axis('off')

            overlay_bi = apply_overlay(img_np, sky_bi[0], color=(1, 0, 0), alpha=0.5)
            axes[1].imshow(overlay_bi)
            axes[1].set_title("Bicubic 'Strict 0' (Sky)\n(Bug: Edges might be lost)")
            axes[1].axis('off')

            overlay_near = apply_overlay(img_np, sky_near[0], color=(0, 1, 0), alpha=0.5)
            axes[2].imshow(overlay_near)
            axes[2].set_title("Nearest 'Strict 0' (Sky)\n(Expected: Sharp Edges)")
            axes[2].axis('off')

            axes[3].imshow(grad_maps[0], cmap='magma')
            axes[3].set_title("Color Gradient Magnitude")
            axes[3].axis('off')

            if has_dynamic:
                overlay_dyn = apply_overlay(img_np, is_dyn[0], color=(0, 0, 1), alpha=0.6)
                axes[4].imshow(overlay_dyn); axes[4].set_title("Dynamic Mask (Objects)"); axes[4].axis('off')

        plt.tight_layout()
        plt.show()

def save_visualization_results(scene_data, indices, output_dir):
    """
    保存渲染结果、Mask叠加图和Alpha合成图。
    
    Args:
        scene_data: 包含推理结果的字典
        indices: 需要保存的帧索引列表 (e.g. [0, 1])
        output_dir: 保存路径
    """
    if scene_data is None:
        print("Error: scene_data is None.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # --- Helpers ---
    def tensor_to_numpy_uint8(tensor):
        # Tensor [C, H, W] -> Numpy [H, W, C] (0-255)
        # Ensure it's detached and on CPU
        arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
        arr = np.clip(arr, 0, 1)
        return (arr * 255).astype(np.uint8)
    
    def get_mask_uint8(tensor):
        # Tensor [C, H, W] -> Numpy [H, W] (0-255)
        # Take first channel
        arr = tensor.detach().cpu().permute(1, 2, 0).numpy()[:, :, 0]
        # Assuming mask is 0-1, 0=Sky, 1=Object
        # We want to visualize "Sky", so let's invert logic for visualization if needed.
        # But usually raw mask visualization is safest.
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def apply_red_overlay(image_uint8, mask_uint8, alpha=0.5):
        """
        image_uint8: [H, W, 3] RGB
        mask_uint8: [H, W] 0-255 (255 will be Red)
        """
        # Create a red layer
        red_layer = np.zeros_like(image_uint8)
        red_layer[:, :, 0] = 255 # R (PIL/Matplotlib order) or B (CV2 order)? 
        # Note: We will save with cv2, which expects BGR. 
        # So Red is channel 2.
        red_layer[:, :, 2] = 255 
        
        # Mask float [0, 1]
        mask_float = mask_uint8.astype(np.float32) / 255.0
        mask_float = np.expand_dims(mask_float, axis=2) # [H, W, 1]
        
        # Blend
        # Region with Mask=1 becomes: (1-alpha)*Img + alpha*Red
        blended = image_uint8.astype(np.float32) * (1 - alpha * mask_float) + \
                  red_layer.astype(np.float32) * (alpha * mask_float)
        
        return np.clip(blended, 0, 255).astype(np.uint8)

    # --- Extract Data ---
    render_imgs = scene_data['rendered_image'] # [T, C, H, W]
    alphas = scene_data['alphas']              # [T, H, W, 1]
    masks = scene_data['masks'][0]             # [T, C, H, W] (Batch 0)
    
    total_frames = render_imgs.shape[0]

    for idx in indices:
        if idx >= total_frames:
            print(f"Skipping index {idx} (out of bounds).")
            continue
            
        print(f"Processing Frame {idx}...")
        
        # 1. Prepare Base Images (RGB)
        # Convert to BGR for OpenCV saving
        img_rgb = tensor_to_numpy_uint8(render_imgs[idx])
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 2. Prepare Alpha
        alpha_map = alphas[idx].detach().cpu().squeeze().numpy() # [H, W] float
        alpha_uint8 = (np.clip(alpha_map, 0, 1) * 255).astype(np.uint8)
        
        # 3. Prepare Sky Mask
        # Raw mask: 0=Sky, 1=Object.
        # We want to highlight SKY in Red. So we want (Mask < Threshold).
        # Let's create a "Is Sky" mask.
        mask_raw = masks[idx].detach().cpu().permute(1, 2, 0).numpy()[:, :, 0] # [H, W] float
        is_sky_mask = (mask_raw < 0.1).astype(np.float32) # 1.0 where Sky, 0.0 where Object
        is_sky_uint8 = (is_sky_mask * 255).astype(np.uint8)
        
        # --- Generate Outputs ---
        
        # A. Predicted Image (Raw)
        path_pred = os.path.join(output_dir, f"frame_{idx:03d}_pred.png")
        cv2.imwrite(path_pred, img_bgr)
        
        # B. Pred + Sky Mask Overlay (Red where code thinks is Sky)
        # 红色区域代表：这里应该是天空。如果红色区域里有物体（比如树梢），说明 Mask 包含了树梢。
        # 如果红色区域很完美，但 Pred Image 在这里有灰色的高斯，那就是 Scale 溢出。
        overlay_sky = apply_red_overlay(img_bgr, is_sky_uint8, alpha=0.4)
        path_sky = os.path.join(output_dir, f"frame_{idx:03d}_overlay_sky.png")
        cv2.imwrite(path_sky, overlay_sky)
        
        # C. Pred + Alpha Composite (Cutout Effect)
        # Simulate: Pixel * Alpha. Background becomes Black.
        # This shows exactly what the Gaussians are "painting".
        alpha_3c = np.stack([alpha_map]*3, axis=-1) # [H, W, 3]
        composite = img_rgb.astype(np.float32) * alpha_3c
        composite_bgr = cv2.cvtColor(np.clip(composite, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        path_comp = os.path.join(output_dir, f"frame_{idx:03d}_composite_alpha.png")
        cv2.imwrite(path_comp, composite_bgr)

    print("All saved.")
    




def apply_heatmap_overlay(image, scalar_map, cmap_name='turbo', alpha=0.5):
    """
    将单通道的 scalar_map (depth/alpha) 归一化并映射为伪彩色，然后叠加在 image 上。
    image: [H, W, 3] numpy, 0-1
    scalar_map: [H, W] numpy
    """
    # Normalize scalar map to 0-1
    norm_map = (scalar_map - scalar_map.min()) / (scalar_map.max() - scalar_map.min() + 1e-8)
    
    # Apply colormap (returns [H, W, 4], RGBA)
    colormap = plt.get_cmap(cmap_name)
    heatmap = colormap(norm_map)[:, :, :3] # Take RGB
    
    # Blend
    output = image * (1 - alpha) + heatmap * alpha
    return np.clip(output, 0, 1)

def visualize_inference_results(scene_data, samples=None, views=1):
    if scene_data is None:
        print("No scene data to visualize.")
        return

    def to_np_img(t):
        return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)
    
    def to_np_mask(t):
        return t.detach().cpu().permute(1, 2, 0).numpy()[:, :, 0]
    
    def _overlay(bg_img, mask, color=(1, 0, 0), alpha=0.5):
        colored = np.zeros_like(bg_img)
        for c in range(3):
            colored[:, :, c] = color[c]
        m = mask[:, :, None]
        return bg_img * (1 - alpha * m) + colored * (alpha * m)

    gt_imgs = scene_data['gt_image']          # [T, C, H, W]
    render_imgs = scene_data['rendered_image'] # [T, C, H, W]
    depths = scene_data['depth_maps']          # [T, H, W]
    alphas = scene_data['alphas']              # [T, H, W, 1]
    masks_bi = scene_data['masks'][0]          # [T, C, H, W]
    masks_near = scene_data['nearest_masks'][0]# [T, C, H, W]
    
    da3_depths = None
    if 'da3_depth' in scene_data and scene_data['da3_depth'] is not None:
        d3 = scene_data['da3_depth']
        if d3.dim() == 5: d3 = d3[0]
        if d3.dim() == 4: d3 = d3.squeeze(1)
        da3_depths = d3

    T_total = gt_imgs.shape[0]
    num_groups = T_total // views if views == 3 else T_total
    
    if samples is None:
        samples = range(num_groups)
    
    print(f"Visualizing {len(list(samples))} groups from total {T_total} frames (views={views})...")

    for g in samples:
        if g >= num_groups:
            print(f"Index {g} out of bounds, skipping.")
            continue

        if views == 3:
            idxs = [3 * g + v for v in range(3)]  # center, left, right
            print(f"Processing Group {g} (frames {idxs})...")

            gts = [to_np_img(gt_imgs[i]) for i in idxs]
            preds = [to_np_img(render_imgs[i]) for i in idxs]
            d_preds = [depths[i].detach().cpu().numpy() for i in idxs]
            alpha_maps = [alphas[i].detach().cpu().squeeze().numpy() for i in idxs]
            sky_bi = [(to_np_mask(masks_bi[i]) < 0.01).astype(float) for i in idxs]
            sky_near = [(to_np_mask(masks_near[i]) < 0.01).astype(float) for i in idxs]

            fig, axs = plt.subplots(6, 1, figsize=(20, 30), dpi=100)
            fig.suptitle(f"Group {g} (Left | Center | Right)", fontsize=16)

            gt_comp = _compose_triview(gts[0], gts[1], gts[2])
            pred_comp = _compose_triview(preds[0], preds[1], preds[2])
            axs[0].imshow(gt_comp); axs[0].set_title("GT"); axs[0].axis('off')
            axs[1].imshow(pred_comp); axs[1].set_title("Predicted"); axs[1].axis('off')

            ov_near = [_overlay(preds[v], sky_near[v], color=(0, 1, 0), alpha=0.5) for v in range(3)]
            axs[2].imshow(_compose_triview(ov_near[0], ov_near[1], ov_near[2]))
            axs[2].set_title("Pred + Nearest Sky (Green)"); axs[2].axis('off')

            d_norms = []
            for d in d_preds:
                d_norms.append((d - d.min()) / (d.max() - d.min() + 1e-8))
            cmap = plt.get_cmap('turbo')
            depth_rgbs = [cmap(dn)[:, :, :3] for dn in d_norms]
            axs[3].imshow(_compose_triview(depth_rgbs[0], depth_rgbs[1], depth_rgbs[2]))
            axs[3].set_title("Predicted Depth"); axs[3].axis('off')

            alpha_rgbs = [np.stack([a]*3, axis=-1) for a in alpha_maps]
            axs[4].imshow(_compose_triview(alpha_rgbs[0], alpha_rgbs[1], alpha_rgbs[2]))
            axs[4].set_title("Opacity (Alpha)"); axs[4].axis('off')

            masked = [preds[v] * np.stack([alpha_maps[v]]*3, axis=-1) for v in range(3)]
            axs[5].imshow(_compose_triview(masked[0], masked[1], masked[2]))
            axs[5].set_title("Predicted * Alpha"); axs[5].axis('off')

        else:
            t = g
            print(f"Processing Frame {t}...")
            img_gt = to_np_img(gt_imgs[t])
            img_pred = to_np_img(render_imgs[t])
            m_bi_raw = to_np_mask(masks_bi[t])
            m_near_raw = to_np_mask(masks_near[t])
            is_sky_bi = (m_bi_raw < 0.01).astype(float)
            is_sky_near = (m_near_raw < 0.01).astype(float)
            d_pred = depths[t].detach().cpu().numpy()
            d_pred_norm = (d_pred - d_pred.min()) / (d_pred.max() - d_pred.min() + 1e-8)
            
            d_da3_norm = None
            if da3_depths is not None:
                d3_val = da3_depths[t].detach().cpu().numpy()
                valid_mask = d3_val > 0
                if valid_mask.any():
                    d_min, d_max = d3_val[valid_mask].min(), d3_val[valid_mask].max()
                    d_da3_norm = (d3_val - d_min) / (d_max - d_min + 1e-8)
                    d_da3_norm[~valid_mask] = 0
                else:
                    d_da3_norm = np.zeros_like(d3_val)

            alpha_map = alphas[t].detach().cpu().squeeze().numpy()

            fig, axs = plt.subplots(4, 2, figsize=(12, 16), dpi=100)
            fig.suptitle(f"Scene Frame {t}", fontsize=16)

            axs[0, 0].imshow(img_gt); axs[0, 0].set_title("GT Image"); axs[0, 0].axis('off')
            axs[0, 1].imshow(img_pred); axs[0, 1].set_title("Predicted Image"); axs[0, 1].axis('off')

            ov_bi = _overlay(img_pred, is_sky_bi, color=(1, 0, 0), alpha=0.5)
            axs[1, 0].imshow(ov_bi)
            axs[1, 0].set_title("Pred + Sky Mask (Bicubic)\nRed = Code thinks is Sky")
            axs[1, 0].axis('off')
            ov_near = _overlay(img_pred, is_sky_near, color=(0, 1, 0), alpha=0.5)
            axs[1, 1].imshow(ov_near)
            axs[1, 1].set_title("Pred + Nearest Sky Mask\nGreen = Strict Sky")
            axs[1, 1].axis('off')

            axs[2, 0].imshow(d_pred_norm, cmap='turbo'); axs[2, 0].set_title("Predicted Depth"); axs[2, 0].axis('off')
            if d_da3_norm is not None:
                axs[2, 1].imshow(d_da3_norm, cmap='turbo'); axs[2, 1].set_title("DA3 Depth Prior")
            else:
                axs[2, 1].text(0.5, 0.5, "DA3 Depth Not Available", ha='center', transform=axs[2, 1].transAxes)
            axs[2, 1].axis('off')

            axs[3, 0].imshow(alpha_map, cmap='gray', vmin=0, vmax=1)
            axs[3, 0].set_title("Opacity (Alpha) Map\nWhite=Opaque, Black=Transparent")
            axs[3, 0].axis('off')
            alpha_3c = np.stack([alpha_map]*3, axis=-1)
            masked_render = img_pred * alpha_3c
            axs[3, 1].imshow(masked_render)
            axs[3, 1].set_title("Predicted Image * Alpha\n(Content from Gaussians only)")
            axs[3, 1].axis('off')

        plt.tight_layout()
        plt.show()



def visualize_depth_alignment_6panel(
    frame_idx,
    dggt_depth, nearest_sky_mask,
    da3_depth_raw, da3_sky_mask,
    final_depth, final_sky_mask,
    output_path=None
):
    """
    可视化：强制抹平天空，只高亮显示物体的几何细节 (Disparity)。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Depth Structure Analysis (Sky Masked & Flattened) - Frame {frame_idx}", fontsize=16)
    
    # --- 1. 核心辅助函数：转视差 + 抹平天空 ---
    def process_for_vis(depth_map, sky_mask):
        # 1. 转视差 (近处值大，远处值小)
        # 加上 1e-6 防止除零
        disp = 1.0 / (depth_map + 1e-6)
        
        # 2. 强制抹平天空 (Sky Mask: 1=Sky, 0=Object)
        # 将天空区域的视差强制设为 0 (或者 NaN)
        # 这样天空就是绝对平滑的单一颜色
        is_sky = (sky_mask > 0.5)
        disp_masked = disp.copy()
        disp_masked[is_sky] = 0 
        
        return disp_masked, is_sky

    # 处理三个深度图
    disp_dggt, sky_dggt = process_for_vis(dggt_depth, nearest_sky_mask)
    disp_da3, sky_da3   = process_for_vis(da3_depth_raw, da3_sky_mask)
    disp_final, sky_final = process_for_vis(final_depth, final_sky_mask)

    # --- 2. 智能归一化 (只统计 Object 区域) ---
    # 我们只用 DGGT 的 Object 区域来确定颜色范围 (vmin, vmax)
    # 这样能保证三张图的颜色标准一致，且对比度最强
    # valid_pixels = disp_dggt[~sky_dggt]
    
    # if len(valid_pixels) > 0:
    #     # 使用 2% - 98% 分位数，切除极端的噪点
    #     vmin = np.percentile(valid_pixels, 2)
    #     vmax = np.percentile(valid_pixels, 98)
    # else:
    #     vmin, vmax = 0, 1
    vmin, vmax = 0, 1

    # --- 3. 绘图 ---
    def plot_map(ax, data, title):
        # 使用 'turbo' 或 'magma'
        # set_bad 可以设置 NaN 的颜色，如果 data 里有 NaN
        im = ax.imshow(data, cmap='turbo', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        return im

    # Row 1: Depth/Disparity Maps (天空已抹平)
    im0 = plot_map(axes[0, 0], disp_dggt, "1. DGGT Disparity (Masked Sky)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im1 = plot_map(axes[0, 1], disp_da3, "3. DA3 Disparity (Masked Sky)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im2 = plot_map(axes[0, 2], disp_final, "5. Final Fused Disparity (Masked Sky)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Masks (显示哪些区域被抹平了)
    def plot_mask(ax, mask, title):
        # mask: 1=Sky, 0=Obj
        # 黑色背景，红色显示天空
        ax.imshow(mask, cmap='gray_r', vmin=0, vmax=1) 
        colored = np.zeros((*mask.shape, 3))
        colored[mask > 0.5] = [1, 0, 0] # Red Sky
        ax.imshow(colored, alpha=0.3)
        ax.set_title(title)
        ax.axis('off')

    plot_mask(axes[1, 0], nearest_sky_mask, "2. DGGT Mask Used")
    plot_mask(axes[1, 1], da3_sky_mask, "4. DA3 Mask Used")
    plot_mask(axes[1, 2], final_sky_mask, "6. Final Mask Used")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


# -----------------------------------------------------------------------------
# Full Sequence Video Saving
# -----------------------------------------------------------------------------

def save_full_sequence_video(images, output_dir, scene_name, logger, fps=8, save_frames=True):
    """
    将完整序列保存为视频和（可选）单独的帧图片。
    
    Args:
        images: [T, C, H, W] tensor, values in [0, 1]
        output_dir: 输出目录
        scene_name: 场景名称
        logger: 日志记录器
        fps: 视频帧率 (默认 8)
        save_frames: 是否保存单独的帧图片 (默认 False，只保存视频)
    
    Returns:
        video_path: 保存的视频路径
    """
    os.makedirs(output_dir, exist_ok=True)
    images_cpu = images.detach().cpu()
    num_frames = images_cpu.shape[0]
    
    image_list = []
    
    # 可选：保存单独的帧图片
    if save_frames:
        frames_dir = os.path.join(output_dir, f"scene_{scene_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        if logger:
            logger.info(f"Saving {num_frames} frames to {frames_dir}...")
        
        for i in range(num_frames):
            rendered = images_cpu[i].clamp(0, 1)
            # 保存单帧图片
            image_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            T.ToPILImage()(rendered).save(image_path)
            # 收集用于视频
            image_list.append(rendered.permute(1, 2, 0).numpy())
    else:
        # 只收集用于视频，不保存单帧
        for i in range(num_frames):
            rendered = images_cpu[i].clamp(0, 1)
            image_list.append(rendered.permute(1, 2, 0).numpy())
    
    # 保存视频
    video_path = os.path.join(output_dir, f"scene_{scene_name}_full_video.mp4")
    if logger:
        logger.info(f"Encoding video with {num_frames} frames at {fps} fps...")
    
    imageio.mimwrite(
        video_path, 
        (np.array(image_list) * 255).astype(np.uint8), 
        fps=fps, 
        codec="libx264"
    )
    
    if save_frames:
        logger.info(f"Saved {num_frames} frames to {frames_dir}")
    logger.info(f"Saved video to {video_path}")
    
    return video_path


def save_comparison_video(gt_images, rendered_images, output_dir, scene_name, logger, fps=8):
    """
    保存 GT 和渲染结果的对比视频（左右并排）。
    
    Args:
        gt_images: [T, C, H, W] tensor, GT images
        rendered_images: [T, C, H, W] tensor, rendered images
        output_dir: 输出目录
        scene_name: 场景名称
        logger: 日志记录器
        fps: 视频帧率
    
    Returns:
        video_path: 保存的视频路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gt_cpu = gt_images.detach().cpu()
    render_cpu = rendered_images.detach().cpu()
    
    num_frames = min(gt_cpu.shape[0], render_cpu.shape[0])
    
    comparison_list = []
    
    logger.info(f"Creating comparison video with {num_frames} frames...")
    
    for i in tqdm(range(num_frames), desc="Creating comparison", unit="frame"):
        gt_frame = gt_cpu[i].clamp(0, 1).permute(1, 2, 0).numpy()
        render_frame = render_cpu[i].clamp(0, 1).permute(1, 2, 0).numpy()
        
        H, W = gt_frame.shape[:2]
        
        # 创建白色分隔条
        separator = np.ones((H, 10, 3), dtype=np.float32)
        
        # 左右并排
        comparison = np.concatenate([gt_frame, separator, render_frame], axis=1)
        comparison_list.append((comparison * 255).astype(np.uint8))
    
    video_path = os.path.join(output_dir, f"scene_{scene_name}_comparison.mp4")
    imageio.mimwrite(video_path, comparison_list, fps=fps, codec="libx264")
    
    logger.info(f"Saved comparison video to {video_path}")
    return video_path