# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as TF
import cv2 
import torch.nn.functional as F
from pathlib import Path
import warnings
from PIL import Image

def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop",return_view="full"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 224

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
        if return_view == "full":
            # return full image without cropping
            img = img 
        elif return_view == "left":
            # return left half (first half horizontally)
            img = img.crop((0, 0, width // 2, height))

        elif return_view == "right":
            # return right half (second half horizontally)
            img = img.crop((width // 2, 0, width, height))
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

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

def load_and_preprocess_videos(video_path_list, mode="crop", target_frames=None, frame_stride=1,return_view='full',start_point=0):
    """
    A quick start function to load and preprocess videos for model input.
    This function handles video loading, frame sampling, and preprocessing.

    Args:
        video_path_list (list): List of paths to video files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.
        target_frames (int, optional): Number of frames to sample from each video. 
                                      If None, uses all frames with frame_stride.
        frame_stride (int, optional): Stride for frame sampling. Default is 1 (use every frame).

    Returns:
        torch.Tensor: Batched tensor of preprocessed video frames with shape (N, T, 3, H, W)

    Raises:
        ValueError: If the input list is empty, if mode is invalid, or if video cannot be loaded

    Notes:
        - Videos with different dimensions will be padded with white (value=1.0)
        - A warning is printed when videos have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        - Frame sampling uses uniform temporal sampling
    """
    # Check for empty list
    if len(video_path_list) == 0:
        raise ValueError("At least 1 video is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    videos = []
    shapes = set()
    target_size = 224

    for video_path in video_path_list:
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        try:
            # Read video metadata to get total frames
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")

            # Determine frames to sample
            if target_frames is None:
                # Use all frames with stride
                frame_indices = list(range(0, total_frames, frame_stride))
                if not frame_indices:
                    frame_indices = [0]  # Fallback to first frame
            else:
                # Uniform temporal sampling
                if target_frames >= total_frames:
                    frame_indices = list(range(total_frames))
                else:
                    frame_indices = torch.linspace(0, total_frames - 1, target_frames).long().tolist()
            from utils.video_utils import load_video_cv2
            video_frames = load_video_cv2(str(video_path), start_point=start_point, frame_skip=frame_stride, num_frames=target_frames)
            video_data = torch.from_numpy(np.stack(video_frames)).permute(0, 3, 1, 2)  # (T, C, H, W)
            # print(f"debug video_data shape {video_data.shape}, total frames {total_frames}, target_frames={target_frames}, sampled frames {len(frame_indices)},dtype {video_data.dtype} range {video_data.min()}-{video_data.max()}")
            if return_view == "full":
                frames = video_data
            elif return_view == "left":
                W = video_data.shape[3]
                frames = video_data[:,:,:,:W//2]
            elif return_view == "right":
                W = video_data.shape[3]
                frames = video_data[:,:,:,W//2:]
                
            # Read specific frames F,C, H, W 
            # debug vggt video data torch.Size([1799, 3, 320, 640]), total frames 1799, sampled frames 1,dtype torch.uint8 range 0-255
            # video_data = read_video(
                # str(video_path), 
                # start_pts=0,
                # end_pts=None,
                # pts_unit='sec', # TODO check this out 2
                # output_format='TCHW'
            # )[0]  # Returns (T, H, W, C), we use TCHW format
            # print(f"debug vggt video data {video_data.shape}, total frames {total_frames}, sampled frames {len(frame_indices)},dtype {video_data.dtype} range {video_data.min()}-{video_data.max()}")
            # Select frames and convert to RGB if needed
            # frames = video_data[frame_indices]  # Shape: (T, H, W, C)
            
            # Convert from [0, 255] to [0, 1] and ensure RGB
            if frames.dtype == torch.uint8:
                frames = frames.float() / 255.0
            
            # Ensure channels are in correct order (T, C, H, W)
            if frames.shape[-1] == 3:  # If format is T H W C
                frames = frames.permute(0, 3, 1, 2)
            elif frames.shape[1] != 3:  # If not RGB
                # Handle grayscale by repeating channels
                if frames.shape[1] == 1:
                    frames = frames.repeat(1, 3, 1, 1)
                else:
                    warnings.warn(f"Unexpected channel format: {frames.shape}, attempting to proceed")

            processed_frames = []
            for frame in frames:
                # Convert tensor to PIL for processing (temporary solution)
                frame_pil = TF.functional.to_pil_image(frame)
                
                width, height = frame_pil.size
                # 960 240 
                # import pdb; pdb.set_trace()
                if mode == "pad":
                    # Make the largest dimension 518px while maintaining aspect ratio
                    if width >= height:
                        new_width = target_size
                        new_height = round(height * (new_width / width) / 14) * 14
                    else:
                        new_height = target_size
                        new_width = round(width * (new_height / height) / 14) * 14
                else:  # mode == "crop"
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14

                # Resize
                frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.BICUBIC)
                frame_tensor = TF.functional.to_tensor(frame_pil)

                # Center crop height if needed (crop mode)
                if mode == "crop" and new_height > target_size:
                    start_y = (new_height - target_size) // 2
                    frame_tensor = frame_tensor[:, start_y:start_y + target_size, :]

                # Padding for pad mode
                if mode == "pad":
                    h_padding = target_size - frame_tensor.shape[1]
                    w_padding = target_size - frame_tensor.shape[2]

                    if h_padding > 0 or w_padding > 0:
                        pad_top = h_padding // 2
                        pad_bottom = h_padding - pad_top
                        pad_left = w_padding // 2
                        pad_right = w_padding - pad_left

                        frame_tensor = F.pad(
                            frame_tensor, (pad_left, pad_right, pad_top, pad_bottom), 
                            mode="constant", value=1.0
                        )

                processed_frames.append(frame_tensor)

            # Stack frames for this video: (T, C, H, W)
            video_tensor = torch.stack(processed_frames)
            shapes.add((video_tensor.shape[2], video_tensor.shape[3]))  # (H, W)
            videos.append(video_tensor)

        except Exception as e:
            raise ValueError(f"Error loading video {video_path}: {str(e)}")

    # Handle different video shapes
    if len(shapes) > 1:
        print(f"Warning: Found videos with different frame shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_videos = []
        for video in videos:
            # Video shape: (T, C, H, W)
            h_padding = max_height - video.shape[2]
            w_padding = max_width - video.shape[3]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad each video: (T, C, H, W) -> pad last two dimensions
                video = F.pad(
                    video, (pad_left, pad_right, pad_top, pad_bottom), 
                    mode="constant", value=1.0
                )
            padded_videos.append(video)
        videos = padded_videos

    # Stack all videos: (N, T, C, H, W)
    videos = torch.stack(videos)

    # Ensure correct shape when single video
    if len(video_path_list) == 1 and videos.dim() == 4:
        videos = videos.unsqueeze(0)

    return videos

