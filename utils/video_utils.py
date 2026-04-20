import cv2 
from typing import List
import numpy as np 
from PIL import Image
import subprocess
import os 

def load_video_cv2(path: str, start_point: int, frame_skip: int, num_frames: int,target_height: int = None , target_width: int = None) -> np.ndarray:
    """
    加载原始视频帧为Numpy数组，形状为 [T, H, W, C], 类型为int8
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    video_real_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_point)
    # print(f"Loading video from {path}, start at frame {start_point}, total {num_frames} frames, skip {frame_skip} fps {cap.get(cv2.CAP_PROP_FPS)}")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            raise ValueError(f"Failed open {path} has {video_real_frame_count} frames at local frame {i}, pos {start_point + i*frame_skip}, actual {actual_pos}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 按需resize; < 1e-3 sec 
        if target_height is not None and target_width is not None:
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame = cv2.resize(frame, (target_width, target_height))
        
        frames.append(frame)  
        
        # 跳过frame_skip-1帧
        if frame_skip > 1 and i < num_frames - 1:
            for _ in range(frame_skip - 1):
                if not cap.grab():
                    raise ValueError(f"Grab failed at skip {i}")
    cap.release()
    return frames

def save_video_cv2(frames: List[np.ndarray], output_video, dst_fps=60):
    # 确保输出是 .mp4
    if not output_video.endswith('.mp4'):
        output_video = output_video + '.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 编码
    dst_h, dst_w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_video, fourcc, dst_fps, (dst_w, dst_h))
    
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_video}")
    
    for frame in frames:
        # 确保 uint8 BGR 格式
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # RGB -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        writer.write(frame)
    
    writer.release()
    return

def concat_videos(video1: List[Image.Image], video2: List[Image.Image],dim='height') -> List[np.ndarray]:
    """将两个视频在指定维度上拼接，返回拼接后的视频帧列表"""
    concatenated_frames = []
    for frame1, frame2 in zip(video1, video2):
        if dim == 'height':
            new_width = max(frame1.width, frame2.width)
            new_height = frame1.height + frame2.height
            new_frame = Image.new('RGB', (new_width, new_height))
            new_frame.paste(frame1, (0, 0))
            new_frame.paste(frame2, (0, frame1.height))
        elif dim == 'width':
            new_width = frame1.width + frame2.width
            new_height = max(frame1.height, frame2.height)
            new_frame = Image.new('RGB', (new_width, new_height))
            new_frame.paste(frame1, (0, 0))
            new_frame.paste(frame2, (frame1.width, 0))
        else:
            raise ValueError("Dimension must be 'height' or 'width'")
        concatenated_frames.append(new_frame)
    return concatenated_frames
