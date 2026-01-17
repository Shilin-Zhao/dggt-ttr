import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
import os
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
import numpy as np


def load_da3_data_from_npz(npz_path, target_image_path, sky_mask_shape):
    """
    Load depth and sky mask from .npz file and match with image file.
    
    Args:
        npz_path: path to .npz file
        target_image_path: path to target image file to match
        sky_mask_shape: (H, W) shape of sky mask for validation
    
    Returns:
        depth: torch.Tensor [H, W] depth map
        sky_mask: torch.Tensor [H, W] sky mask (bool, True=sky)
        matched: bool, whether the npz file matches the image file
    """
    data = np.load(npz_path)
    
    # Get source filename from npz
    # Convert numpy array to string if needed
    source_filename_raw = data["source_filename"]
    if isinstance(source_filename_raw, np.ndarray):
        # Handle numpy array: use item() for scalar, or convert to string
        if source_filename_raw.ndim == 0:
            source_filename = str(source_filename_raw.item())
        else:
            source_filename = str(source_filename_raw)
    else:
        source_filename = str(source_filename_raw)
    source_filename = os.path.basename(source_filename)
    target_filename = os.path.basename(target_image_path)
    
    # Check if filenames match (without extension)
    source_name = os.path.splitext(source_filename)[0]
    target_name = os.path.splitext(target_filename)[0]
    
    if source_name != target_name:
        return None, None, False
    
    # Load depth and sky
    depth = data["depth"]  # [H, W] float32
    sky = data["sky"]      # [H, W] bool
    
    # Assert shape matches sky_mask_shape
    assert depth.shape == sky_mask_shape, f"DA3 depth shape {depth.shape} does not match sky_mask shape {sky_mask_shape}"
    assert sky.shape == sky_mask_shape, f"DA3 sky shape {sky.shape} does not match sky_mask shape {sky_mask_shape}"
    
    # Convert to torch tensors
    depth_tensor = torch.from_numpy(depth).float()
    sky_tensor = torch.from_numpy(sky).bool()
    
    return depth_tensor, sky_tensor, True


def load_da3_data_for_indices(da3_result_paths, image_paths, indices, sky_mask_shape, views=1):
    """
    Load DA3 data for given indices.
    
    Args:
        da3_result_paths: list of npz file paths
        image_paths: list of image paths (or list of lists for views=3)
        indices: list of indices to load
        sky_mask_shape: (H, W) shape of sky mask
        views: number of views (1 or 3)
    
    Returns:
        da3_depth: torch.Tensor [S, 1, H, W] or [S*3, 1, H, W]
        da3_sky_mask: torch.Tensor [S, 1, H, W] or [S*3, 1, H, W]
        success: bool, whether all data was loaded successfully
    """
    da3_depths = []
    da3_sky_masks = []
    
    if views == 1:
        for i in indices:
            image_path = image_paths[i]
            matched = False
            for npz_path in da3_result_paths:
                depth, sky_mask, matched = load_da3_data_from_npz(
                    npz_path, image_path, sky_mask_shape
                )
                if matched:
                    da3_depths.append(depth)
                    da3_sky_masks.append(sky_mask)
                    break
            if not matched:
                print(f"Warning: No matching DA3 npz file for {os.path.basename(image_path)}")
                return None, None, False
    else:  # views == 3
        for i in indices:
            for v in range(3):
                image_path = image_paths[v][i]
                matched = False
                for npz_path in da3_result_paths:
                    depth, sky_mask, matched = load_da3_data_from_npz(
                        npz_path, image_path, sky_mask_shape
                    )
                    if matched:
                        da3_depths.append(depth)
                        da3_sky_masks.append(sky_mask)
                        break
                if not matched:
                    print(f"Warning: No matching DA3 npz file for {os.path.basename(image_path)}")
                    return None, None, False
    
    # Stack tensors
    if len(da3_depths) > 0:
        da3_depth = torch.stack(da3_depths, dim=0)  # [S, H, W] or [S*3, H, W]
        da3_sky_mask = torch.stack(da3_sky_masks, dim=0)  # [S, H, W] or [S*3, H, W]
        # Add channel dimension
        da3_depth = da3_depth.unsqueeze(1)  # [S, 1, H, W] or [S*3, 1, H, W]
        da3_sky_mask = da3_sky_mask.unsqueeze(1).float()  # [S, 1, H, W] or [S*3, 1, H, W]
        return da3_depth, da3_sky_mask, True
    
    return None, None, False


def get_filename_without_ext(filepath):
    """Extract filename without extension for comparison."""
    return os.path.splitext(os.path.basename(filepath))[0]


def verify_filenames_match(image_paths, da3_paths, data_type="da3_depth"):
    """Verify that da3 file names match image file names."""
    if len(image_paths) != len(da3_paths):
        print(f"Warning: {data_type} file count mismatch: images={len(image_paths)}, {data_type}={len(da3_paths)}")
        return False
    
    mismatches = []
    for img_path, da3_path in zip(image_paths, da3_paths):
        img_name = get_filename_without_ext(img_path)
        da3_name = get_filename_without_ext(da3_path)
        if img_name != da3_name:
            mismatches.append((img_path, da3_path, img_name, da3_name))
    
    if mismatches:
        print(f"Warning: {data_type} filename mismatches detected:")
        for img_path, da3_path, img_name, da3_name in mismatches[:5]:  # Print first 5 mismatches
            print(f"  Image: {os.path.basename(img_path)} -> {data_type}: {os.path.basename(da3_path)}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more mismatches")
        return False
    
    return True

def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()

    
def load_and_preprocess_flow(flow_path_list, extrinsic_paths, intrinsic_path, height, width):
    if len(flow_path_list) == 0:
        raise ValueError("At least 1 image is required")

    flows = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # print(f"[DEBUG] flow_path_list: {flow_path_list}")
    for i, flow_path in enumerate(flow_path_list):
        # print(f"[DEBUG] Processing flow_path[{i}]: {flow_path}")
        # print(f"[DEBUG] Is file: {os.path.isfile(flow_path) if flow_path else 'Empty/None'}")
        # print(f"[DEBUG] Is dir: {os.path.isdir(flow_path) if flow_path else 'Empty/None'}")
        depth_and_flow = np.load(flow_path)
        flow = depth_and_flow
        flow = torch.tensor(flow).float()
        flow = resize_flow(flow, (height, width))
        flows.append(flow)
    
    return torch.stack(flows)


def load_and_preprocess_images(image_path_list, mode="crop", resample_method=Image.Resampling.BICUBIC):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), resample_method)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    images = torch.stack(images)  # concatenate images
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


class WaymoOpenDataset(Dataset):
    def __init__(self, image_dir, scene_names = None, sequence_length= None, start_idx = -1, mode=1, views=1, intervals=2):
        #mode 1 : train
        #mode 2 : pure reconstruction
        #mode 3 : interplation
        
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        if mode == 1:
            interval = 1
        elif mode == 2:
            interval = 1
        elif mode == 3:
            interval = intervals
        else:
            interval = 1
        self.interval =  interval
        self.mode = mode
        if mode == 1:
            test_mode = False
            load_flow = False
        elif mode == 2:
            test_mode = True
            load_flow = False
        elif mode == 3:
            test_mode = True
            load_flow = True
        else:
            pass
        self.test_mode = test_mode
        self.load_flow = load_flow
        self.views = views

        # Scan all scene folders and collect image paths
        if scene_names is None:
            scene_names = [] 
            scene_names_ = [f"{i:03d}" for i in range(0, 99)]
            scene_names = scene_names + scene_names_
        self.scenes = scene_names
        self.image_paths = []
        self.sky_mask_paths = []
        self.dynamic_mask_path = []
        self.extrinsic_paths = []
        self.intrinsic_paths = []
        self.semantic_mask_path = []
        self.depth_flow_paths = []
        self.ego_paths = []
        self.da3_result_paths = []  # Store paths to .npz files in da3_result folder

        self.start_idx = start_idx

        for scene_name in scene_names:
            scene_path = os.path.join(image_dir, scene_name, "images")
            if os.path.isdir(scene_path):
                # image
                if self.views == 1:
                    image_paths = sorted(
                        [
                            os.path.join(scene_path, f)
                            for f in os.listdir(scene_path)
                            if f.endswith(("_0.jpg", "_0.png"))
                        ]
                    )
                    self.image_paths.append(image_paths)
                elif self.views == 3:
                    views_image_lists = []
                    for v in range(3):
                        suffixes = (f"_{v}.jpg", f"_{v}.png")
                        files_v = sorted(
                            [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(suffixes)]
                        )
                        views_image_lists.append(files_v)
                    lengths = [len(l) for l in views_image_lists]
                    if len(set(lengths)) != 1:
                        raise RuntimeError(f"Inconsistent number of images across views in scene {scene_name}, lengths: {lengths}")
                    self.image_paths.append(views_image_lists)

                # sky_mask
                sky_mask_path = os.path.join(image_dir, scene_name, "sky_masks")
                if os.path.isdir(sky_mask_path):
                    if self.views == 1:
                        sky_mask_paths = sorted(
                            [os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.sky_mask_paths.append(sky_mask_paths)
                    elif self.views == 3:
                        views_sky_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(suffixes)])
                            views_sky_lists.append(files_v)
                        self.sky_mask_paths.append(views_sky_lists)
                else:
                    self.sky_mask_paths.append([] if self.views == 1 else [[] for _ in range(3)])

                # extrinsic
                extrinsic_path = os.path.join(image_dir, scene_name, "ego_pose")
                if os.path.isdir(extrinsic_path):
                    extrinsic_paths = sorted([
                        os.path.join(extrinsic_path, f)
                        for f in os.listdir(extrinsic_path)
                        if f.endswith(".txt")
                    ])
                    self.extrinsic_paths.append(extrinsic_paths)
                else:
                    self.extrinsic_paths.append([])

                # extrinsic
                ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                # ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                if os.path.isdir(ego_path):
                    ego_path =  os.path.join(ego_path, "0.txt")
                    self.ego_paths.append(ego_path)

                # intrinsic
                intrinsic_path = os.path.join(image_dir, scene_name, "intrinsics")
                if os.path.isdir(intrinsic_path):
                    if self.views == 1:
                        intrinsic_paths = os.path.join(intrinsic_path, "0.txt")
                        self.intrinsic_paths.append(intrinsic_paths)
                    elif self.views == 3:
                        intrinsics_views = []
                        for v in range(3):
                            p = os.path.join(intrinsic_path, f"{v}.txt")
                            intrinsics_views.append(p if os.path.exists(p) else "")
                        self.intrinsic_paths.append(intrinsics_views)
                else:
                    self.intrinsic_paths.append("" if self.views == 1 else ["" for _ in range(3)])

                # dynamic mask
                dynamic_mask_path = os.path.join(image_dir, scene_name, "fine_dynamic_masks/all")
                if os.path.isdir(dynamic_mask_path):
                    if self.views == 1:
                        dynamic_mask_paths = sorted(
                            [os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.dynamic_mask_path.append(dynamic_mask_paths)
                    elif self.views == 3:
                        views_dyn_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(suffixes)])
                            views_dyn_lists.append(files_v)
                        self.dynamic_mask_path.append(views_dyn_lists)
                else:
                    self.dynamic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])
                # depth
                depth_path = os.path.join(image_dir, scene_name, "depth_flows_4")
                if os.path.isdir(depth_path):
                    if self.views == 1:
                        depth_paths = sorted(
                            [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith("_0.npy")]
                        )
                        self.depth_flow_paths.append(depth_paths)
                    elif self.views == 3:
                        views_depth_lists = []
                        for v in range(3):
                            suffix = f"_{v}.npy"
                            files_v = sorted(
                                [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith(suffix)]
                            )
                            views_depth_lists.append(files_v)
                        self.depth_flow_paths.append(views_depth_lists)
                else:
                    self.depth_flow_paths.append([] if self.views == 1 else [[] for _ in range(3)])
                # semantic mask
                semantic_mask_path = os.path.join(image_dir, scene_name, "custom_masks")
                if os.path.isdir(semantic_mask_path):
                    if self.views == 1:
                        semantic_mask_paths = sorted(
                            [os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.semantic_mask_path.append(semantic_mask_paths)
                    elif self.views == 3:
                        views_sem_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(suffixes)])
                            views_sem_lists.append(files_v)
                        self.semantic_mask_path.append(views_sem_lists)
                else:
                    self.semantic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])
                
                # da3_result (npz files containing depth and sky mask)
                da3_result_path = os.path.join(image_dir, scene_name, "da3_result")
                if os.path.isdir(da3_result_path):
                    # Get all .npz files
                    npz_files = sorted([f for f in os.listdir(da3_result_path) if f.endswith(".npz")])
                    if self.views == 1:
                        # For views=1, we need to match npz files with image files by source_filename
                        # We'll store the npz file paths and match them in __getitem__
                        da3_result_paths = [os.path.join(da3_result_path, f) for f in npz_files]
                        self.da3_result_paths.append(da3_result_paths)
                    elif self.views == 3:
                        # For views=3, we need to organize by view
                        # We'll need to match by source_filename in __getitem__
                        da3_result_paths = [os.path.join(da3_result_path, f) for f in npz_files]
                        self.da3_result_paths.append(da3_result_paths)
                else:
                    self.da3_result_paths.append([])


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        sky_mask_paths = self.sky_mask_paths[idx]
        dynamic_mask_paths = self.dynamic_mask_path[idx]
        semantic_mask_paths = self.semantic_mask_path[idx]
        da3_result_paths = self.da3_result_paths[idx]

        start_idx = random.randint(0, max(1, len(image_paths[0] if self.views == 3 else image_paths) - 21))

        if self.mode == 1:
            indices = [start_idx]
            intervals = sorted(random.sample(range(1, 20), self.sequence_length - 1))
            for interval in intervals:
                indices.append(start_idx + interval)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
                nearest_masks = load_and_preprocess_images(mask_seq, resample_method=Image.Resampling.NEAREST)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]
                nearest_masks = load_and_preprocess_images(mask_seq, resample_method=Image.Resampling.NEAREST)  # [S*3, C, H, W]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)

            input_dict = {
                "images": images,
                "masks": masks,
                "nearest_masks": nearest_masks,
                "image_paths": seq,
                "timestamps": timestamps,
                "interval": intervals,
            }

        
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            # da3_result (load from .npz files)
            if len(da3_result_paths) > 0:
                da3_depths = []
                da3_sky_masks = []
                
                if self.views == 1:
                    for i in indices:
                        image_path = image_paths[i]
                        # Find matching npz file
                        matched = False
                        for npz_path in da3_result_paths:
                            depth, sky_mask, matched = load_da3_data_from_npz(
                                npz_path, image_path, (sky_mask_h, sky_mask_w)
                            )
                            if matched:
                                da3_depths.append(depth)
                                da3_sky_masks.append(sky_mask)
                                break
                        if not matched:
                            print(f"Warning: No matching DA3 npz file for {os.path.basename(image_path)}")
                    
                    if len(da3_depths) == len(indices):
                        # Stack to [S, H, W]
                        da3_depth = torch.stack(da3_depths, dim=0)  # [S, H, W]
                        da3_sky_mask = torch.stack(da3_sky_masks, dim=0)  # [S, H, W]
                        # Add channel dimension for consistency: [S, 1, H, W]
                        da3_depth = da3_depth.unsqueeze(1)  # [S, 1, H, W]
                        da3_sky_mask = da3_sky_mask.unsqueeze(1).float()  # [S, 1, H, W]
                        input_dict["da3_depth"] = da3_depth
                        input_dict["da3_sky_mask"] = da3_sky_mask
                        
                elif self.views == 3:
                    for i in indices:
                        for v in range(3):
                            image_path = image_paths[v][i]
                            # Find matching npz file
                            matched = False
                            for npz_path in da3_result_paths:
                                depth, sky_mask, matched = load_da3_data_from_npz(
                                    npz_path, image_path, (sky_mask_h, sky_mask_w)
                                )
                                if matched:
                                    da3_depths.append(depth)
                                    da3_sky_masks.append(sky_mask)
                                    break
                            if not matched:
                                print(f"Warning: No matching DA3 npz file for {os.path.basename(image_path)}")
                    
                    if len(da3_depths) == len(indices) * 3:
                        # Stack to [S*3, H, W]
                        da3_depth = torch.stack(da3_depths, dim=0)  # [S*3, H, W]
                        da3_sky_mask = torch.stack(da3_sky_masks, dim=0)  # [S*3, H, W]
                        # Add channel dimension: [S*3, 1, H, W]
                        da3_depth = da3_depth.unsqueeze(1)  # [S*3, 1, H, W]
                        da3_sky_mask = da3_sky_mask.unsqueeze(1).float()  # [S*3, 1, H, W]
                        input_dict["da3_depth"] = da3_depth
                        input_dict["da3_sky_mask"] = da3_sky_mask
            
            # if len(semantic_mask_paths) > 0:
            #     if self.views == 1:
            #         sem_mask_seq = [semantic_mask_paths[i] for i in indices]
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S, C, H, W]
            #     elif self.views == 3:
            #         sem_mask_seq = []
            #         for i in indices:
            #             for v in range(3):
            #                 sem_mask_seq.append(semantic_mask_paths[v][i])
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S*3, C, H, W]
            #     semantic_mask = semantic_mask * 255 / 10
            #     semantic_mask = semantic_mask.int()
            #     semantic_mask[semantic_mask > 9] = 255
            #     input_dict["semantic_mask"] = semantic_mask

            return input_dict

        elif self.mode == 2: 
            # honor the provided start_idx when doing reconstruction
            start_idx = self.start_idx if self.start_idx is not None and self.start_idx >= 0 else 0
            # ensure we do not go out of bounds
            seq_len = len(image_paths[0]) if self.views == 3 else len(image_paths)
            max_start = max(0, seq_len - self.sequence_length * self.interval)
            start_idx = min(start_idx, max_start)
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
                nearest_masks = load_and_preprocess_images(mask_seq, resample_method=Image.Resampling.NEAREST)  # [S, C, H, W]
                sky_mask_h, sky_mask_w = nearest_masks.shape[2], nearest_masks.shape[3]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]
                nearest_masks = load_and_preprocess_images(mask_seq, resample_method=Image.Resampling.NEAREST)  # [S*3, C, H, W]
                sky_mask_h, sky_mask_w = nearest_masks.shape[2], nearest_masks.shape[3]
                


            input_dict = {
                "images": images,
                "image_paths": seq,
                "masks": masks,
                "nearest_masks": nearest_masks,
                "timestamps": timestamps,
                "interval": intervals,
            }
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                else:
                    depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = depth_data

            # da3_result (load from .npz files)
            if len(da3_result_paths) > 0:
                da3_depth, da3_sky_mask, success = load_da3_data_for_indices(
                    da3_result_paths, image_paths, indices, (sky_mask_h, sky_mask_w), self.views
                )
                if success:
                    input_dict["da3_depth"] = da3_depth
                    input_dict["da3_sky_mask"] = da3_sky_mask

            return input_dict

        else:  # self.mode == 3
            start_idx = self.start_idx if self.start_idx is not None and self.start_idx >= 0 else 0
            # ensure we do not go out of bounds
            seq_len = len(image_paths[0]) if self.views == 3 else len(image_paths)
            max_start = max(0, seq_len - self.sequence_length * self.interval)
            start_idx = min(start_idx, max_start)
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            target_indices = [start_idx + i for i in range(self.sequence_length * self.interval - (self.interval - 1))]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
            
            # images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
                target_seq = [image_paths[i] for i in target_indices]
                target_images = load_and_preprocess_images(target_seq)  # [T, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

                target_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_seq.append(image_paths[v][i])
                target_images = load_and_preprocess_images(target_seq)  # [T*3, C, H, W]

            # sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
                nearest_masks = load_and_preprocess_images(mask_seq, resample_method=Image.Resampling.NEAREST)  # [S, C, H, W]
                target_mask_seq = [sky_mask_paths[i] for i in target_indices]
                target_masks = load_and_preprocess_images(target_mask_seq)  # [T, C, H, W]
                target_nearest_masks = load_and_preprocess_images(target_mask_seq, resample_method=Image.Resampling.NEAREST)  # [T, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]
                nearest_masks = load_and_preprocess_images(mask_seq, resample_method=Image.Resampling.NEAREST)  # [S*3, C, H, W]

                target_mask_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_mask_seq.append(sky_mask_paths[v][i])
                target_masks = load_and_preprocess_images(target_mask_seq)  # [T*3, C, H, W]
                target_nearest_masks = load_and_preprocess_images(target_mask_seq, resample_method=Image.Resampling.NEAREST)  # [T*3, C, H, W]

            input_dict = {
                "images": images,
                "targets": target_images,
                "masks": masks,
                "nearest_masks": nearest_masks,
                "image_paths": seq,
                "timestamps": timestamps,
                # "target_timestamps": target_timestamps,
                "interval": intervals,
                "target_masks": target_masks,
                "target_nearest_masks": target_nearest_masks,
            }

            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                    target_dy_mask_seq = [dynamic_mask_paths[i] for i in target_indices]
                    target_dynamic_mask = load_and_preprocess_images(target_dy_mask_seq)  # [T, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    target_dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    for i in target_indices:
                        for v in range(3):
                            target_dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)         # [S*3, C, H, W]
                    target_dynamic_mask = load_and_preprocess_images(target_dy_mask_seq)  # [T*3, C, H, W]
                input_dict["dynamic_mask"] = target_dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_seq = [self.depth_flow_paths[idx][i] for i in target_indices]
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        target_depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        for i in target_indices:
                            for v in range(3):
                                target_depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                else:
                    target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = target_depth_data

            # da3_result (load from .npz files, use target_indices for mode 3)
            if len(da3_result_paths) > 0:
                # Get sky mask shape from target masks
                if self.views == 1:
                    target_mask_seq = [sky_mask_paths[i] for i in target_indices]
                    target_nearest_masks = load_and_preprocess_images(target_mask_seq, resample_method=Image.Resampling.NEAREST)
                    target_sky_mask_h, target_sky_mask_w = target_nearest_masks.shape[2], target_nearest_masks.shape[3]
                else:
                    target_mask_seq = []
                    for i in target_indices:
                        for v in range(3):
                            target_mask_seq.append(sky_mask_paths[v][i])
                    target_nearest_masks = load_and_preprocess_images(target_mask_seq, resample_method=Image.Resampling.NEAREST)
                    target_sky_mask_h, target_sky_mask_w = target_nearest_masks.shape[2], target_nearest_masks.shape[3]
                
                da3_depth, da3_sky_mask, success = load_da3_data_for_indices(
                    da3_result_paths, image_paths, target_indices, (target_sky_mask_h, target_sky_mask_w), self.views
                )
                if success:
                    input_dict["da3_depth"] = da3_depth
                    input_dict["da3_sky_mask"] = da3_sky_mask

            return input_dict
