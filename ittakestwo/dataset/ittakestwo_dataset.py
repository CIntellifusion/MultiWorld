import functools
import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange

def choice_blocks(rng: np.random.Generator,
                  high: int,        # per-sample upper bound m+1
                  size: int,        # total number of samples to draw
                  dtype=np.intp):
    """Return a 1-D array of length=size, block-shuffled, no repeats within a block."""
    if size <= high:                 # one pass is enough
        return rng.choice(high, size, replace=False)

    full, rem = divmod(size, high)
    # Create full rounds of 0..high-1, shuffle each round individually
    base = np.arange(high, dtype=dtype).repeat(full).reshape(full, high)
    for row in base:
        rng.shuffle(row)
    base = base.ravel()
    
    # Final partial round
    if rem:
        base = np.concatenate([base, rng.choice(high, rem, replace=False)])
    return base


class ItTakesTwoBaseDataset(Dataset):
    """
    ItTakesTwo dataset base class, encapsulating shared logic:
    - Video/action parameter parsing
    - Action loading (CSV -> Numpy/Torch)
    - Video frame loading and format conversion
    - View processing (left/right split)
    - Gamepad to keyboard action conversion
    - Lazy action loading (do not preload during initialization)
    """
    
    # Unified action space keys (10 discrete + 2 continuous per player)
    # Output format: [F, 2, 10] for discrete, [F, 2, 2] for continuous
    # Player 0 (index 0): Left player (keyboard)
    # Player 1 (index 1): Right player (gamepad converted to keyboard)
    UNIFIED_DISCRETE_KEYS = ["w", "a", "s", "d", "space", "shift", "ctrl", "e", "q", "f"]
    UNIFIED_CONTINUOUS_KEYS = ["look_x", "look_y"]
    
    def __init__(
        self,
        video_params: Dict[str, Any],
        action_config: Optional[Dict[str, Any]] = None,
        load_action: bool = True,
        return_video_type: str = "pil",  # "pil", "tensor", "numpy"
        return_view: str = "both",       # "both", "left", "right", "random"
        frame_processor: Optional[Callable] = None,
        convert_gamepad_to_keyboard: bool = False,  # Convert right player (gamepad) actions to keyboard actions
        lazy_load_action: bool = False,  # Do not preload actions at init; read them in __getitem__
        stick_threshold: float = 0.3,  # Threshold for stick-to-WASD conversion
    ):
        self.video_params = video_params
        self.action_config = action_config or {}
        self.load_action = load_action
        self.return_video_type = return_video_type
        self.return_view = return_view
        self.frame_processor = frame_processor
        self.convert_gamepad_to_keyboard = convert_gamepad_to_keyboard
        self.lazy_load_action = lazy_load_action
        self.stick_threshold = stick_threshold
        
        # Parse video parameters
        self.frame_skip = video_params.get('frame_skip', 1)
        self.target_height = video_params.get('height', 320)
        self.target_width = video_params.get('width', 640)
        self.num_frames = video_params.get('num_frames', 81)
        
        # Parse action config
        self.discrete_action_keys = self.action_config.get('discrete_action_keys', [])
        self.continuous_action_keys = self.action_config.get('continuous_action_keys', [])
        
        # If conversion is enabled, update action keys to unified action space
        if self.convert_gamepad_to_keyboard:
            self.discrete_action_keys = self.UNIFIED_DISCRETE_KEYS
            self.continuous_action_keys = self.UNIFIED_CONTINUOUS_KEYS
    
    @functools.cached_property
    def usecols(self) -> List[str]:
        """Return the action column names to load."""
        return self.discrete_action_keys + self.continuous_action_keys
    
    @functools.cached_property
    def left_player_keys(self) -> List[str]:
        """Left player action keys."""
        discrete = ["a", "s", "d", "space", "shift", "ctrl", "e", "q", "f"]
        continuous = ["norm_dx", "norm_dy"]
        return discrete + continuous
    
    @functools.cached_property
    def right_player_keys(self) -> List[str]:
        """Right player action keys."""
        discrete = [f"button_{i}" for i in range(9)]
        continuous = ["axis_0", "axis_1", "axis_2", "axis_3"]
        return discrete + continuous
    
    def _preload_action_to_numpy(
        self, 
        action_path: str, 
        mask_action_keys: Optional[List[str]] = None,
        use_original_keys: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Convert CSV to a Numpy dictionary in one pass.
        Supports masking (zero out specified keys, used for left/right player separation).
        Supports gamepad-to-keyboard conversion (when convert_gamepad_to_keyboard=True).
        
        Args:
            action_path: Path to the action CSV file
            mask_action_keys: List of keys to mask
            use_original_keys: Whether to use original keys (for lazy loading raw CSV)
        """
        mask_action_keys = mask_action_keys or []
        
        # Determine which keys to read from CSV
        if use_original_keys or not self.convert_gamepad_to_keyboard:
            discrete_keys = self.action_config.get('discrete_action_keys', [])
            continuous_keys = self.action_config.get('continuous_action_keys', [])
        else:
            # In conversion mode, read all original keys
            discrete_keys = self.action_config.get('discrete_action_keys', [])
            continuous_keys = self.action_config.get('continuous_action_keys', [])
        
        usecols = discrete_keys + continuous_keys
        
        try:
            df = pd.read_csv(action_path, usecols=usecols)
        except Exception as e:
            print(f"action_path {action_path} {usecols}, error: {e}")
            raise
        
        n_frames = len(df)
        
        # Apply mask
        for key in mask_action_keys:
            if key in df.columns:
                df[key] = 0 if key in discrete_keys else 0.0
        
        result = {}
        
        # Convert gamepad -> keyboard if needed
        if self.convert_gamepad_to_keyboard:
            discrete_data, continuous_data = self._convert_gamepad_to_unified(df, n_frames)
            result['discrete_action'] = discrete_data
            result['continuous_action'] = continuous_data
        else:
            # Discrete actions
            if discrete_keys:
                discrete_data = np.zeros((n_frames, len(discrete_keys)), dtype=np.int64)
                for i, key in enumerate(discrete_keys):
                    if key in df.columns:
                        discrete_data[:, i] = df[key].to_numpy(dtype=np.int64)
                result['discrete_action'] = discrete_data
            
            # Continuous actions
            if continuous_keys:
                continuous_data = np.zeros((n_frames, len(continuous_keys)), dtype=np.float32)
                for i, key in enumerate(continuous_keys):
                    if key in df.columns:
                        continuous_data[:, i] = df[key].to_numpy(dtype=np.float32)
                result['continuous_action'] = continuous_data
        
        del df
        return result
    
    def _convert_gamepad_to_unified(self, df: pd.DataFrame, n_frames: int) -> tuple:
        """
        Convert gamepad actions to unified keyboard action space, output shape is dual-player separated [F, 2, D].
        
        Args:
            df: DataFrame containing raw actions
            n_frames: Number of frames
            
        Returns:
            (discrete_action, continuous_action): Converted numpy arrays
                      discrete: [F, 2, 10] - 2 players, 10 discrete each
                      continuous: [F, 2, 2] - 2 players, 2 continuous each
        """
        # Initialize action space: [F, 2, D] where 2 is num_players
        discrete_data = np.zeros((n_frames, 2, 10), dtype=np.int64)  # [F, 2, 10]
        continuous_data = np.zeros((n_frames, 2, 2), dtype=np.float32)  # [F, 2, 2]
        
        # ==================== Left player (keyboard + mouse) - player_idx=0 ====================
        # WASD (indices 0-3)
        if "w" in df.columns:
            discrete_data[:, 0, 0] = df["w"].to_numpy(dtype=np.int64)
        if "a" in df.columns:
            discrete_data[:, 0, 1] = df["a"].to_numpy(dtype=np.int64)
        if "s" in df.columns:
            discrete_data[:, 0, 2] = df["s"].to_numpy(dtype=np.int64)
        if "d" in df.columns:
            discrete_data[:, 0, 3] = df["d"].to_numpy(dtype=np.int64)
        
        # Left player buttons (indices 4-9)
        if "space" in df.columns:
            discrete_data[:, 0, 4] = df["space"].to_numpy(dtype=np.int64)
        if "shift" in df.columns:
            discrete_data[:, 0, 5] = df["shift"].to_numpy(dtype=np.int64)
        if "ctrl" in df.columns:
            discrete_data[:, 0, 6] = df["ctrl"].to_numpy(dtype=np.int64)
        if "e" in df.columns:
            discrete_data[:, 0, 7] = df["e"].to_numpy(dtype=np.int64)
        if "q" in df.columns:
            discrete_data[:, 0, 8] = df["q"].to_numpy(dtype=np.int64)
        if "f" in df.columns:
            discrete_data[:, 0, 9] = df["f"].to_numpy(dtype=np.int64)
        
        # Left player camera (continuous indices 0-1)
        if "norm_dx" in df.columns:
            continuous_data[:, 0, 0] = df["norm_dx"].to_numpy(dtype=np.float32)
        if "norm_dy" in df.columns:
            continuous_data[:, 0, 1] = df["norm_dy"].to_numpy(dtype=np.float32)
        
        # ==================== Right player (gamepad) - player_idx=1 ====================
        # Gamepad buttons -> keyboard buttons (mapped to right player)
        if "button_0" in df.columns:  # Jump (A) -> space (index 4)
            discrete_data[:, 1, 4] = df["button_0"].to_numpy(dtype=np.int64)
        if "button_1" in df.columns:  # Crouch (B) -> ctrl (index 6)
            discrete_data[:, 1, 6] = df["button_1"].to_numpy(dtype=np.int64)
        if "button_2" in df.columns:  # Dash (X) -> shift (index 5)
            discrete_data[:, 1, 5] = df["button_2"].to_numpy(dtype=np.int64)
        if "button_3" in df.columns:  # Ability (Y) -> f (index 9)
            discrete_data[:, 1, 9] = df["button_3"].to_numpy(dtype=np.int64)
        if "button_4" in df.columns:  # Aim (LB) -> q (index 8)
            discrete_data[:, 1, 8] = df["button_4"].to_numpy(dtype=np.int64)
        if "button_5" in df.columns:  # Throw (RB) -> e (index 7)
            discrete_data[:, 1, 7] = df["button_5"].to_numpy(dtype=np.int64)
        if "button_8" in df.columns:  # Sprint (L3) -> shift (index 5)
            discrete_data[:, 1, 5] = np.maximum(discrete_data[:, 1, 5], df["button_8"].to_numpy(dtype=np.int64))
        # button_9 is unused and dropped
        
        # Left stick -> WASD (thresholded to binary)
        if "axis_0" in df.columns:  # Left stick X -> a/d (indices 1, 3)
            axis_0 = df["axis_0"].to_numpy(dtype=np.float32)
            discrete_data[:, 1, 1] = (axis_0 < -self.stick_threshold).astype(np.int64)  # a
            discrete_data[:, 1, 3] = (axis_0 > self.stick_threshold).astype(np.int64)  # d
        
        if "axis_1" in df.columns:  # Left stick Y -> w/s (indices 0, 2)
            axis_1 = df["axis_1"].to_numpy(dtype=np.float32)
            discrete_data[:, 1, 0] = (axis_1 < -self.stick_threshold).astype(np.int64)  # w
            discrete_data[:, 1, 2] = (axis_1 > self.stick_threshold).astype(np.int64)  # s
        
        # Right stick -> look_x, look_y (continuous indices 0-1)
        if "axis_2" in df.columns:  # Right stick X -> look_x
            continuous_data[:, 1, 0] = df["axis_2"].to_numpy(dtype=np.float32)
        if "axis_3" in df.columns:  # Right stick Y -> look_y
            continuous_data[:, 1, 1] = df["axis_3"].to_numpy(dtype=np.float32)
        
        return discrete_data, continuous_data
    
    def _process_return_view(self, frame: np.ndarray, return_view: str) -> np.ndarray:
        """
        Process single-frame view according to return_view.
        Input: [H, W, C]
        Output: [H, W/2, C] or [H, W, C]
        """
        if return_view == "left":
            frame = frame[:, :self.target_width // 2, :]
        elif return_view == "right":
            frame = frame[:, self.target_width // 2:, :]
        return frame
    
    def _load_raw_video(
        self, 
        path: str, 
        start_point: int, 
        frame_skip: int,
        return_view: str,
        num_frames: int, 
    ) -> List[np.ndarray]:
        """
        Load raw video frames as a list of Numpy arrays.
        Shape: [T, H, W, C], dtype uint8.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_point)
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.release()
                raise ValueError(
                    f"Failed at frame {i}, expected pos {start_point + i*frame_skip}, "
                    f"actual {actual_pos} in {path}"
                )
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # View cropping
            frame = self._process_return_view(frame, return_view)
            frames.append(frame)
            
            # Skip frames
            if frame_skip > 1 and i < num_frames - 1:
                for _ in range(frame_skip - 1):
                    if not cap.grab():
                        cap.release()
                        raise ValueError(f"Grab failed at skip frame {i}")
        
        cap.release()
        return frames 
    
    def _convert_frames(
        self, 
        frames: List[np.ndarray],
        return_video_type: str, 
    ) -> Union[torch.Tensor, List[Image.Image], np.ndarray]:
        """
        Convert a list of numpy frames to the specified output format.
        """
        if return_video_type == "tensor":
            # [T,H,W,C] -> [B,C,T,H,W], B=1
            tensor = torch.from_numpy(np.array(frames, dtype=np.float32))
            tensor = tensor.permute(3, 0, 1, 2)[None, ...]
            tensor = tensor / 127.5 - 1.0  # Normalize to [-1, 1]
            return tensor.to(torch.bfloat16)
        
        elif return_video_type == "pil":
            pil_frames = [Image.fromarray(frame) for frame in frames]
            if self.frame_processor is not None:
                pil_frames = self.frame_processor(pil_frames)
            return pil_frames
        
        elif return_video_type == "numpy":
            return np.array(frames, dtype=np.uint8)
        
        else:
            raise ValueError(f"Unknown return_video_type: {self.return_video_type}")
    
    def _load_video(
        self, 
        path: str, 
        start: int, 
        frame_skip: int,
        return_view: str,
        num_frames: int,
    ) -> Union[torch.Tensor, List[Image.Image], np.ndarray]:
        """Load video and convert to the specified format."""
        frames = self._load_raw_video(path, start, frame_skip, return_view,num_frames)
        return self._convert_frames(frames,self.return_video_type)
    
    def _load_action_from_numpy(
        self, 
        action_numpy: Dict[str, np.ndarray], 
        indices: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Load actions from a numpy dict and return torch tensors (with batch dimension)."""
        result = {}
        for key in ['discrete_action', 'continuous_action']:
            if key in action_numpy:
                result[key] = torch.from_numpy(action_numpy[key][indices])[None, ...]
        return result
    def _load_split_action(
        self, 
        action_numpy: Dict[str, np.ndarray], 
        indices: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Load actions with left/right player separation."""
        # Lazy loading handling
        if self.lazy_load_action and 'action_path' in action_numpy:
            action_path = action_numpy['action_path']
            # Read and convert actions on-the-fly in __getitem__
            action_numpy = self._preload_action_to_numpy(action_path, use_original_keys=True)
        
        base = self._load_action_from_numpy(action_numpy, indices)
        
        if 'left_player_action' in action_numpy:
            base['left_player_action'] = self._load_action_from_numpy(
                action_numpy['left_player_action'], indices
            )
        if 'right_player_action' in action_numpy:
            base['right_player_action'] = self._load_action_from_numpy(
                action_numpy['right_player_action'], indices
            )
        return base
    def action_from_array_to_df(self, action_array: np.ndarray) -> pd.DataFrame:
        """Convert an action array to a DataFrame."""
        return pd.DataFrame(action_array, columns=self.usecols)
    
    def action_from_array_to_dict(self, action_array: np.ndarray) -> Dict[str, float]:
        """Convert an action array to a dictionary."""
        return {key: action_array[i] for i, key in enumerate(self.usecols)}


class IttakestwoVideoActionDataset(ItTakesTwoBaseDataset):
    """
    Full video dataset supporting metadata CSV, random start points, left/right view separation, and action masking.
    Supports gamepad-to-keyboard action conversion and lazy loading.
    """
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        video_params: Dict[str, Any],
        action_config: Optional[Dict[str, Any]] = None,
        load_action: bool = True,
        share_action: bool = True,
        repeat: int = 1,
        load_from_cache: bool = False,
        data_file_keys: Optional[List[str]] = None,
        random_start_point: bool = False,
        frame_processor: Optional[Callable] = None,
        return_video_type: str = "pil",
        return_view: str = "both",
        max_entries: Optional[int] = None,
        convert_gamepad_to_keyboard: bool = False,
        lazy_load_action: bool = False,
        stick_threshold: float = 0.3,
    ):
        super().__init__(
            video_params=video_params,
            action_config=action_config,
            load_action=load_action,
            return_video_type=return_video_type,
            return_view=return_view,
            frame_processor=frame_processor,
            convert_gamepad_to_keyboard=convert_gamepad_to_keyboard,
            lazy_load_action=lazy_load_action,
            stick_threshold=stick_threshold,
        )
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.share_action = share_action
        self.repeat = repeat
        self.load_from_cache = load_from_cache
        self.data_file_keys = data_file_keys or ["video", "action"]
        self.random_start_point = random_start_point
        if return_view == "random" and max_entries is not None: 
            max_entries = max_entries * 2 # Each video generates two samples (left/right views), so double the limit
        self.max_entries = max_entries
        
        self._preload_metadata()
        self.samples = self.samples * self.repeat
        self.start_points = self._precompute_start_points()
    
    def _preload_metadata(self):
        """Preload video metadata and actions."""
        print("Preloading video metadata...")
        df = pd.read_csv(self.metadata_path)
        self.video_metadata = {}
        self.samples = []
        required_span = self.num_frames * self.frame_skip
        num_entry = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            video_name = row['video']
            action_name = row['action']
            video_path = os.path.join(self.base_path, video_name)
            action_path = os.path.join(self.base_path, action_name)
            total_frames = row['frame_count']
            
            if total_frames < required_span:
                print(f"Warning: {video_name} has {total_frames} frames < {required_span}, skipping")
                continue
            
            self.video_metadata[video_name] = {'total_frames': total_frames}
            
            if not os.path.exists(video_path) or not os.path.exists(action_path):
                continue
            
            # Preload or lazy-load actions
            if self.lazy_load_action:
                # Lazy load: only store the path, do not load data
                action_numpy = {"action_path": action_path}
            else:
                # Preload actions
                action_numpy = self._preload_action_to_numpy(action_path)
                if not self.convert_gamepad_to_keyboard:
                    # Only split left/right player actions when not in conversion mode
                    left_action = self._preload_action_to_numpy(action_path, mask_action_keys=self.right_player_keys)
                    right_action = self._preload_action_to_numpy(action_path, mask_action_keys=self.left_player_keys)
                    action_numpy['left_player_action'] = left_action
                    action_numpy['right_player_action'] = right_action
            if self.return_view == "random":
                # Create independent samples for left/right views
                left_sample = {
                    "video": video_path,
                    "video_name": video_name,
                    "action": action_numpy if self.share_action else left_action,
                    "return_view": "left",
                }
                right_sample = {
                    "video": video_path,
                    "video_name": video_name,
                    "action": action_numpy if self.share_action else right_action,
                    "return_view": "right",
                }
                self.samples.extend([left_sample, right_sample])
                num_entry += 2
            else:
                self.samples.append({
                    "video": video_path,
                    "video_name": video_name,
                    "action": action_numpy,
                    "return_view": self.return_view,
                })
                num_entry += 1
            
            if self.max_entries and num_entry >= self.max_entries:
                print(f"Reached max_entries: {self.max_entries}")
                break
        
        self.num_original_samples = num_entry
    
    def _precompute_start_points(self) -> np.ndarray:
        """Precompute start points for each sample to ensure reproducibility."""
        global_rank = int(os.environ.get("RANK", 0))
        rng = np.random.RandomState(42 + global_rank)
        base_samples = self.samples[:self.num_original_samples]
        span = self.num_frames * self.frame_skip
        
        max_starts = np.array([
            self.video_metadata[s['video_name']]['total_frames'] - span
            for s in base_samples
        ])
        
        if not self.random_start_point:
            # Uniform sampling
            intervals = (max_starts + 1) / self.repeat
            starts = np.floor(
                np.arange(self.repeat)[None, :] * intervals[:, None] + intervals[:, None] / 2
            ).astype(np.int64)
            starts = starts.T.ravel()
        else:
            # Random block sampling
            starts = np.vstack([
                choice_blocks(rng, m + 1, self.repeat, dtype=np.int64)
                for m in max_starts
            ])
            starts = starts.T.ravel()
        
        return starts
    
    def _get_sampling_indices(self, video_length: int, sample_idx: int) -> np.ndarray:
        """Get sampling indices."""
        span = self.num_frames * self.frame_skip
        if video_length < span:
            raise ValueError(f"Video too short: {video_length} < {span}")
        start = self.start_points[sample_idx]
        indices = start + np.arange(self.num_frames) * self.frame_skip
        return indices.astype(np.int64)
    
    def __len__(self):
        return len(self.samples)
    def load_env_obv(self,sample,data,frame_index=0):
        """
        return shape: B,F,K,C,H,W 
        """
        if "env_obv" in self.data_file_keys or "env_obv_dino" in self.data_file_keys: 
            from diffsynth.models.vggt.utils.load_fn import load_and_preprocess_videos
            left_view = load_and_preprocess_videos(
                [sample['video']], mode="pad", target_frames=1, frame_stride=1,return_view="left",start_point=frame_index
            )[None,...]
            right_view = load_and_preprocess_videos(
                [sample['video']], mode="pad", target_frames=1, frame_stride=1, return_view="right",start_point=frame_index
            )[None,...]
            return torch.cat([left_view,right_view],dim=2)
        elif "env_obv_vae" in self.data_file_keys: 
            initial_image = self._load_raw_video(sample['video'], frame_index, self.frame_skip, return_view="both",num_frames=1)
            initial_image = self._convert_frames(initial_image,return_video_type='tensor') # B,C,F,H,W
            W = initial_image.shape[-1]
            left_view = initial_image[:,:,:,:,W//2:][None,...] #1,B,C,T,H,W
            right_view = initial_image[:,:,:,:,:W//2][None,...]
            concat_view = torch.cat([left_view,right_view],dim=0) # K,B,C,T,H,W
            concat_view = rearrange(concat_view,"k b c t h w -> b k c t h w")
            return concat_view
        else:
            return None 
        
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        video_length = self.video_metadata[sample['video_name']]['total_frames']
        indices = self._get_sampling_indices(video_length, idx)
        
        data = {
            'real_idx': idx,
            'start_point': indices[0],
            'video_name': sample['video_name'],
        }
        
        if 'video' in self.data_file_keys:
            data['video'] = self._load_video(
                sample['video'], indices[0], self.frame_skip, sample['return_view'],self.num_frames,
            )
        
        if self.load_action:
            data['action'] = self._load_split_action(sample['action'], indices)
    
        data['env_obv'] = self.load_env_obv(sample,data,frame_index = indices[0])
        
        return data


class IttakestwoVideoDataset(ItTakesTwoBaseDataset):
    """
    Simplified video dataset: loads directly from video/action directories, no random start points.
    Suitable for testing or simple scenarios.
    """
    def __init__(
        self,
        video_dir: str,
        action_dir: str,
        video_params: Dict[str, Any],
        action_config: Optional[Dict[str, Any]] = None,
        load_action: bool = False,
        data_file_keys: Optional[List[str]] = None,
        return_video_type: str = "pil",
        return_view: str = "both",
        frame_processor: Optional[Callable] = None,
        convert_gamepad_to_keyboard: bool = False,
        lazy_load_action: bool = False,
        stick_threshold: float = 0.3,
    ):
        # Image datasets usually only need single-frame parameters
        super().__init__(
            video_params=video_params,
            action_config=action_config,
            load_action=load_action,
            return_video_type=return_video_type,
            return_view=return_view,
            frame_processor=frame_processor,
            convert_gamepad_to_keyboard=convert_gamepad_to_keyboard,
            lazy_load_action=lazy_load_action,
            stick_threshold=stick_threshold,
        )
        self.video_dir = video_dir
        self.action_dir = action_dir
        self.data_file_keys = data_file_keys or ["video", "action"]
        self._load_samples()
    
    def _load_samples(self):
        """Scan directories and load samples."""
        video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])
        action_files = set(os.listdir(self.action_dir))
        
        self.samples = []
        for vf in video_files:
            af = vf.replace('.mp4', '.csv')
            if af not in action_files:
                raise ValueError(f"Missing action file for {vf}")
            
            video_path = os.path.join(self.video_dir, vf)
            action_path = os.path.join(self.action_dir, af)
            
            if self.lazy_load_action:
                # Lazy load: only store the path
                action_numpy = {"action_path": action_path}
            else:
                action_numpy = self._preload_action_to_numpy(action_path)
            
            self.samples.append({
                "video": video_path,
                "video_name": vf,
                "action": action_numpy,
            })
    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        # Simple sequential sampling starting from frame 0
        indices = np.arange(self.num_frames) * self.frame_skip
        
        data = {
            'real_idx': idx,
            'start_point': 0,
            'video_name': sample['video_name'],
        }
        
        if 'video' in self.data_file_keys:
            data['video'] = self._load_video(
                sample['video'], 0, self.frame_skip, self.return_view, self.num_frames
            )
        
        if self.load_action:
            # Lazy loading handling
            if self.lazy_load_action and 'action_path' in sample['action']:
                action_numpy = self._preload_action_to_numpy(sample['action']['action_path'], use_original_keys=True)
                data['action'] = self._load_action_from_numpy(action_numpy, indices)
            else:
                data['action'] = self._load_action_from_numpy(sample['action'], indices)
        

        return data


class IttakestwoImageActionDataset(ItTakesTwoBaseDataset):
    """
    Image-action dataset: for evaluation, supports Cartesian product of images and actions.
    """
    def __init__(
        self,
        base_path: str,
        image_meta_path: str,
        action_meta_path: str,
        video_params: Dict[str, Any],
        action_config: Dict[str, Any],
        return_view: str = "both",
        return_video_type: str = "tensor",
        frame_processor: Optional[Callable] = None,
        convert_gamepad_to_keyboard: bool = False,
        lazy_load_action: bool = False,
        stick_threshold: float = 0.3,
    ):
        # Image datasets usually only need single-frame parameters
        super().__init__(
            video_params=video_params,
            action_config=action_config,
            load_action=True,  # Image datasets must load actions
            return_video_type=return_video_type,
            return_view=return_view,
            frame_processor=frame_processor,
            convert_gamepad_to_keyboard=convert_gamepad_to_keyboard,
            lazy_load_action=lazy_load_action,
            stick_threshold=stick_threshold,
        )
        self.base_path = Path(base_path)
        
        # Load metadata
        img_df = pd.read_csv(image_meta_path)
        self.img_names = img_df["video"].astype(str).str.strip().tolist()
        self.action_df = pd.read_csv(action_meta_path)
        
        # Preload all actions
        self.action_list = self._build_action_cache()
        
        # Cartesian product: each image paired with each action
        self.cart = [(i, a) for i in range(len(self.img_names)) 
                    for a in range(len(self.action_df))]
        
        print(f"[ImageDataset] {len(self.img_names)} imgs x {len(self.action_df)} "
              f"acts = {len(self)} samples")
    
    def _build_action_cache(self) -> List[Dict[str, torch.Tensor]]:
        """Preload or lazy-load all actions."""
        cache = []
        for _, row in self.action_df.iterrows():
            action_path = str(self.base_path / str(row['action']).strip())
            
            if self.lazy_load_action:
                # Lazy load: only store the path
                cache.append({"action_path": action_path})
            else:
                action_numpy = self._preload_action_to_numpy(action_path)
                if not self.convert_gamepad_to_keyboard:
                    left_action = self._preload_action_to_numpy(action_path, mask_action_keys=self.right_player_keys)
                    right_action = self._preload_action_to_numpy(action_path, mask_action_keys=self.left_player_keys)
                    action_numpy['left_player_action'] = left_action
                    action_numpy['right_player_action'] = right_action
                cache.append(action_numpy)
        return cache
    
    def _load_single_image(
        self, 
        path: str
    ) -> Union[torch.Tensor, List[Image.Image], np.ndarray]:
        """Load a single image and convert to the specified format."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        
        # Resize
        img = img.resize((self.target_width, self.target_height), Image.LANCZOS)
        # View processing
        if self.return_view == "left":
            img = img.crop((0, 0, w // 2, h))
        elif self.return_view == "right":
            img = img.crop((w // 2, 0, w, h))
        elif self.return_view == "random":
            img = img.crop((0, 0, w // 2, h)) if np.random.rand() < 0.5 else img.crop((w // 2, 0, w, h))
        
        print(img)
        # Format conversion (consistent with video interface)
        if self.return_video_type == "tensor":
            arr = np.array(img, dtype=np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
            tensor = tensor / 127.5 - 1.0
            tensor = tensor.to(torch.bfloat16)
            return tensor[None, :, None, ...]  # [1,C,1,H,W]
        
        elif self.return_video_type == "pil":
            if self.frame_processor:
                img = self.frame_processor([img])[0]
            return [img]  # List[Image]
        
        elif self.return_video_type == "numpy":
            return np.array(img, dtype=np.uint8)[None, ...]  # [1,H,W,C]
        else:
            raise ValueError(f"Unknown type: {self.return_video_type}")
    
    def __len__(self):
        return len(self.cart)
    
    def load_env_obv_image(self,image_path):
        from diffsynth.models.vggt.utils.load_fn import load_and_preprocess_images
        left_view = load_and_preprocess_images(
            [image_path], mode="pad",return_view="left"
        )[None,None,...] #  (1,1,3, H, W)
        right_view = load_and_preprocess_images(
            [image_path], mode="pad",return_view="right"
        )[None,None,...] #  (1,1,3, H, W)
        env_obv =  torch.cat([left_view,right_view],dim=2) # b,f,k,c,h,w
        # print(f"debug env obv shape {env_obv.shape}")
        return env_obv
    def __getitem__(self, idx):
        img_idx, act_idx = self.cart[idx]
        img_path = self.base_path / self.img_names[img_idx]
        action_data = self.action_list[act_idx]
        
        # Image-action datasets usually take frame-0 actions (or single-frame actions)
        print(self._load_split_action(
                action_data, np.arange(self.num_frames)).keys())
        return {
            "video": self._load_single_image(str(img_path)),
            "env_obv": self.load_env_obv_image(str(img_path)),
            "video_name": self.img_names[img_idx],
            "action": self._load_split_action(
                action_data, np.arange(self.num_frames)  # Single-frame action
            ),
        }
