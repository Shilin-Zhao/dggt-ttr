<div align="center">

# DGGT-TTR: Robust 4D Reconstruction with Test-Time Refinement

**A robust fork of DGGT featuring precise mask processing and multi-stage test-time optimization.**

[Original Paper](https://arxiv.org/abs/2512.03004) | [Original Project Page](https://xiaomi-research.github.io/dggt/)

</div>

## 🚀 Key Enhancements

This repository improves upon the original DGGT implementation by addressing segmentation artifacts and introducing a Test-Time Refinement (TTR) pipeline.

### 1. Fix: Vanishing Small Objects

**Problem:** The original implementation uses interpolation methods to load sky masks, which converts discrete mask values into continuous values. Subsequently, the code incorrectly treats these continuous values as discrete when computing `(sky_mask == 0).any(dim=-1)`, causing some objects (especially distant thin structures like street lamps) to be incorrectly identified as sky and disappear intermittently.

**Solution:** We use nearest-neighbor interpolation (`Image.Resampling.NEAREST`) to preserve the discrete nature of mask values, ensuring stable object visibility across frames.

### 2. Feature: Test-Time Refinement (TTR)

We introduce a multi-stage optimization pipeline that runs during inference (using the input video itself) to align the 3D Gaussians with the observed images.

**Stage 1: Pose Optimization**  
Refines camera trajectories to better match the observed scene geometry.

**Stage 2: Attribute Optimization**  
Optimizes Gaussian Splatting parameters including:
- Opacity, color, and rotation (always enabled)
- Position (XYZ) and scale (optional, may cause overfitting)

**Note:** While optimizing position and scale achieves the largest quantitative improvements, it may lead to overfitting and artifacts in novel view synthesis (NVS). We recommend validating on held-out viewpoints when using full optimization.

## Quantitative Results

| Configuration | PSNR | SSIM | LPIPS | Improvement (PSNR) |
|--------------|------|------|-------|-------------------|
| Baseline (Original DGGT) | 28.9651 | 0.9055 | 0.0885 | - |
| + Nearest Mask Interpolation | 29.1532 | 0.9063 | 0.0870 | +0.1881 |
| + Pose Refinement | 29.2365 | 0.9092 | 0.0863 | +0.2714 |
| + Pose + GS (no XYZ/scale) | 30.8060 | 0.9278 | 0.0834 | +1.8408 |
| + Pose + GS (full) | 31.6438 | 0.9340 | 0.0769 | +2.6786 |

**Results on Scene 001 (20 frames):**
- **Baseline**: PSNR=28.9651, SSIM=0.9055, LPIPS=0.0885
- **With nearest interpolation**: PSNR=29.1532, SSIM=0.9063, LPIPS=0.0870
- **Pose refinement only**: PSNR=29.2365, SSIM=0.9092, LPIPS=0.0863
- **Pose + GS (no position/scale)**: PSNR=30.8060, SSIM=0.9278, LPIPS=0.0834
- **Pose + GS (full optimization)**: PSNR=31.6438, SSIM=0.9340, LPIPS=0.0769

**⚠️ Overfitting Warning:** Full optimization (including XYZ and scales) shows the largest improvements but may overfit to training views. Validation on novel viewpoints is recommended.

## Visual Comparison

*Visual comparison videos will be added here showing ground truth, baseline, and TTR results side-by-side.*

## Installation

For installation instructions, please refer to the [original DGGT repository](https://github.com/xiaomi-research/dggt).

Quick setup:
```bash
conda create -n dggt python=3.10
conda activate dggt
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
cd third_party/pointops2 && python setup.py install && cd ../..
```

Download checkpoints from the [original repository](https://github.com/xiaomi-research/dggt).

## Usage

### Basic Inference (with nearest mask interpolation)
```bash
python inference.py \
    --image_dir /path/to/images \
    --scene_names 001 \
    --input_views 1 \
    --sequence_length 20 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/checkpoint.pth \
    --output_path /path/to/output \
    -images \
    -metrics \
    -diffusion
```

### Test-Time Refinement
```bash
python inference_refine.py \
    --image_dir /path/to/images \
    --scene_names 001 \
    --input_views 1 \
    --sequence_length 20 \
    --start_idx 0 \
    --ckpt_path /path/to/checkpoint.pth \
    --output_path /path/to/output \
    --enable_refinement \
    --refine_pose \
    --refine_gs \
    --refine_xyz \
    --refine_steps_pose 50 \
    --refine_steps_gs 50 \
    -images \
    -metrics \
    -diffusion
```

**Refinement Parameters:**
- `--enable_refinement`: Enable test-time refinement
- `--refine_pose`: Optimize camera poses
- `--refine_gs`: Optimize GS attributes (opacity, color, rotation)
- `--refine_xyz`: Optimize GS positions (may cause overfitting)
- `--refine_steps_pose`: Optimization steps for pose (default: 50)
- `--refine_steps_gs`: Optimization steps for GS (default: 50)

## Roadmap

### Current Progress
- ✅ Fixed sky mask interpolation issue
- ✅ Implemented pose refinement
- ✅ Implemented GS attribute refinement
- ✅ Added support for selective parameter optimization

### Next Steps
- [ ] Integrate [Depth Anything 3](https://depth-anything-3.github.io/) as depth prior for improved geometry
- [ ] Novel view synthesis validation to assess overfitting
- [ ] Multi-scene evaluation and benchmarking

## Acknowledgement & Citation

This project is a fork of DGGT. I express my gratitude to the original authors for their excellent research and open-source contribution.

**Original Paper:** [arXiv:2512.03004](https://arxiv.org/abs/2512.03004)

**Original Repository:** [https://github.com/xiaomi-research/dggt](https://github.com/xiaomi-research/dggt)

If you use this work, please cite the original DGGT paper:

```bibtex
@article{chenfeedforward,
  title={Feedforward 4D Reconstruction for Dynamic Driving Scenes using Unposed Images},
  author={Chen, Xiaoxue and Xiong, Ziyi and Chen, Yuantao and Li, Gen and Wang, Nan and Luo, Hongcheng and Chen, Long and Sun, Haiyang and WANG, BING and Chen, Guang and others}
}
```

## License

This project is licensed under the Apache License 2.0.

Some files in this repository are derived from VGGT (facebookresearch/vggt) and are licensed under the VGGT upstream license. See NOTICE for details.
