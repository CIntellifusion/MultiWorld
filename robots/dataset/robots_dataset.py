import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
from diffsynth.models.vggt.utils.load_fn import load_and_preprocess_videos
from einops import rearrange
from pathlib import Path
import numpy as np  
import json 
import random 
from scipy.signal import savgol_filter
import cv2 
### Refactored for env-action-video dataset 
def get_num_frames(reader,time_division_factor=4, time_division_remainder=1, num_frames=81):
    num_frames_local = num_frames
    total_frames = int(reader.count_frames())
    
    if total_frames < num_frames_local:
        num_frames_local = total_frames
        while num_frames_local > 1 and num_frames_local % time_division_factor != time_division_remainder:
            num_frames_local -= 1
    
    return num_frames_local, total_frames
def get_random_start_point(path):
    reader = imageio.get_reader(path)
    num_frames_actual, total_frames = get_num_frames(reader)
    # Determine start frame
    if total_frames > num_frames_actual:
        start_frame = random.randint(0, total_frames - num_frames_actual)
    else:
        start_frame = 0
        
    # Ensure no out-of-bounds access
    if start_frame + num_frames_actual > total_frames:
        start_frame = max(0, total_frames - num_frames_actual)
    return start_frame 

class LoadActionDataV3:
    def __init__(self,action_max,action_min,num_frames=81,temporal_downsample=4,action_type='pos'):
        self.num_frames = num_frames
        self.temporal_downsample = temporal_downsample
        self.action_max = np.array(action_max)[None,None,:]
        self.action_min = np.array(action_min)[None,None,:]
        self.action_type = action_type
        
        fps = 30 
        self.fps = fps
        self.dt = 1.0 / fps
        self.smooth_window = 7 
    def __call__(self, file_path,start_point):
        # load global action 
        end_point = start_point + self.num_frames if self.num_frames is not None else None
        action_data = np.load(file_path)[:,start_point:end_point] 
        # round up for temporal downsample 
        num_frames = action_data.shape[1] 
        # indices = [0] + [i for i in range(1,num_frames,self.temporal_downsample)]
        need_round_up = num_frames % self.temporal_downsample != 1 # wan vae 1+4temporal compression 
        last_indice = [num_frames -1] if need_round_up else []
        indices = [i for i in range(0,num_frames,self.temporal_downsample)] + last_indice
        # print(f"debug indices {indices} for action data with num_frames {num_frames}")
        action_data = action_data[:,indices]
        if self.action_type == 'vel':
            # not limited to [-1,1]
            action_data = self.convert_to_vel(action_data)
        action_data = (action_data - self.action_min) / (self.action_max - self.action_min) * 2.0 -1.0  # normalize to -1~1
        action_data = torch.from_numpy(action_data).float()[None,...] #unsequenzer for batch dimension
        # print(f"debug action data shape {action_data.shape} action_data [{action_data.min().item():.2f},{action_data.max().item():.2f}]")
        return action_data
    
    # =========================================================================
    # Core conversion 1: position → velocity (for pd_joint_vel controller)
    # =========================================================================

    def convert_to_vel(self, action_pos_gripper):
        """
        Convert (B, N, 8) action data → (B, N, 8)
        - Frame 0: joint position (rad)
        - Frames 1…N-1: joint velocity (rad/s)
        - Gripper state appended back unchanged
        Supports B ≥ 1 (multi-agent)
        """
        B, N, _ = action_pos_gripper.shape
        dt = self.dt

        # Output container
        action_vel_full = np.empty_like(action_pos_gripper)   # (B, N, 8)

        for b in range(B):                     # Process per agent
            # 1. Extract 7 joint positions
            q = action_pos_gripper[b, :, :7]   # (N, 7)

            # 2. Compute velocity
            window_length = min(self.smooth_window,
                                N if N % 2 == 1 else N - 1)
            vel = np.empty_like(q)             # (N, 7)

            if window_length >= 3:
                for j in range(7):
                    vel[:, j] = savgol_filter(q[:, j],
                                            window_length=window_length,
                                            polyorder=3,
                                            deriv=1,
                                            delta=dt)
            else:                              # Data too short
                for j in range(7):
                    vel[:, j] = np.gradient(q[:, j], dt)

            # 3. First frame keeps position, rest are velocities
            out = np.empty_like(q)             # (N, 7)
            out[0, :] = q[0, :]                # rad
            if N > 1:
                out[1:, :] = vel[1:, :]        # rad/s

            # 4. Append gripper state back
            gripper = action_pos_gripper[b, :, 7:]  # (N, 1)
            action_vel_full[b] = np.concatenate([out, gripper], axis=-1)

        return action_vel_full

class RobotsSimulationDataset(torch.utils.data.Dataset):
    def __init__(self, 
                # data config 
                base_path,
                metadata_path,
                video_params,
                return_video_type,
                random_start_point,
                # multiview config
                load_env_obv,
                env_obv_type, 
                num_views,
                num_camera_pose_frames,
                load_action,
                max_agent, 
                action_type=None,
                action_min=None,
                action_max=None,
                # dataset config 
                repeat=1,
                data_file_keys=tuple("video"),
                output_dir=None,
                load_from_cache=False,
                max_entries=None, 
                temporal_downsample=4,
                sep_action=False,
                **kwargs,
    ):
        # Dataset for multi-agent simulator
        super(RobotsSimulationDataset).__init__()
        print(f"warning no kwargs used but got : {kwargs.keys()}")
        self.base_path = base_path
        self.metadata_path = metadata_path
        assert self.base_path is not None, "base_path should not be None" 
        assert self.metadata_path is not None, "metadata_path should not be None"
        
        self.max_entries = max_entries if max_entries is not None else 999999999
        self.random_start_point = random_start_point
        
        self.load_env_obv = load_env_obv
        self.env_obv_type = env_obv_type
        self.num_views = num_views
        
        self.load_action = load_action
        self.action_type = action_type
        self.max_agent = max_agent
        self.action_min = action_min
        self.action_max = action_max 
        self.num_camera_pose_frames = num_camera_pose_frames 
        self.temporal_downsample = temporal_downsample 
        
        self.video_params = video_params
        self.frame_skip = video_params.get('frame_skip', 1)
        self.target_height = video_params.get('height', 320)
        self.target_width = video_params.get('width', 640)
        self.num_frames = video_params.get('num_frames', 81)
        self.return_video_type = return_video_type 
        self.sep_action = sep_action
        print(f"RobotsSimulationDataset initialized with random_start={self.random_start_point}, num_frames={self.num_frames}")
        self.repeat = repeat
        self.load_from_cache = load_from_cache
        self.data_file_keys = data_file_keys
        self.output_dir = output_dir 
        self.current_video_info = None
        self.load_metadata(metadata_path)
        self.contruct_observation_relationship(output_dir)
        self.frame_processor = None 
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def get_unique_key(self,data):
        video_key = "video"
        full_video_path = self.rel_path_to_full(data[video_key], self.base_path)
        episode_id = os.path.basename(full_video_path).split("_")[0]
        chunk_id = os.path.basename(os.path.dirname(full_video_path))
        task_name = data.get("prompt", "unknown_task")
        unique_key = f"{task_name}_{chunk_id}_{episode_id}"
        return unique_key
    def contruct_observation_relationship(self,output_dir=None):
        # videos from one episode are related. 
        # data is a list of dict with keys : ["video", "prompt", "action_state_latents", "camera_pose","num_frames","num_agents"]
        # the unique key here is taskname+episodeid 
        self.episode_to_data_ids = {}
        for idx, data_entry in enumerate(self.data):
            video_path = data_entry.get("video", "")
            unique_key = self.get_unique_key(data_entry)
            if unique_key not in self.episode_to_data_ids:
                self.episode_to_data_ids[unique_key] = []
            self.episode_to_data_ids[unique_key].append(video_path)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir,"episode_to_data_ids.json")
        with open(save_path, "w") as f:
            json.dump(self.episode_to_data_ids, f, indent=4)
        print(f"Saved episode to data ids mapping to {save_path}")
        # create video_path to camera mapping 
        self.video_path_to_camera = {}
        for idx, data_entry in enumerate(self.data):
            video_path = data_entry.get("video", "")
            camera_path = data_entry.get("camera_pose", "")
            self.video_path_to_camera[video_path] = camera_path
            # print(f"Mapping video {video_path} to camera {camera_path}")
            
    def rel_path_to_full(self, rel_path,base_path=None):
        assert rel_path is not None, "rel_path should not be None"
        if base_path is None:
            base_path = self.base_path
        full_path = os.path.join(base_path, rel_path)
        assert os.path.exists(full_path), f"Path {full_path} does not exist."
        return full_path
    
    def load_camera_pose(self,camera_path):
        full_camera_path= self.rel_path_to_full(camera_path, self.base_path) # this should be relative to dataset base 
        camera_data = np.load(full_camera_path) # [1,1,16]; [b,f,16]
        camera_poses = torch.from_numpy(camera_data).float()  # Convert to torch tensor
        fx,fy,cx,cy = camera_poses[:,:,:4][0][0]
        camera_poses[:,:,:4] = torch.Tensor([fx/cx,fy/cy,fx/cx,fy/cy]) # 1/tan(fov/2)
        F = self.num_camera_pose_frames
        camera_poses = camera_poses.repeat(1, F, 1)  # (1, F, 16)
        return camera_poses.contiguous()
    def _load_raw_video(self, path: str, start_point: int, frame_skip: int) -> np.ndarray:
        """
        Load raw video frames as Numpy array with shape [T, H, W, C] and dtype uint8.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_point)
        if self.num_frames is None: 
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames_to_read=total_frames - start_point
        else:
            num_frames_to_read=self.num_frames
        for i in range(num_frames_to_read):
            ret, frame = cap.read()
            if not ret:
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                raise ValueError(f"Path {path} Failed at frame {i}, pos {start_point + i*frame_skip}, actual {actual_pos}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize on demand; < 1e-3 sec
            if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            frames.append(frame)  
            
            # Skip frame_skip-1 frames
            if frame_skip > 1 and i < self.num_frames - 1:
                for _ in range(frame_skip - 1):
                    if not cap.grab():
                        raise ValueError(f"Grab failed at skip {i}")
        cap.release()
        return frames

    def _load_video(self, path: str, start: int, frame_skip: int,return_video_type=None):
        """Load video frames and return in the specified format."""
        frames = self._load_raw_video(path, start, frame_skip)  # List of np.ndarray [H,W,C]
        if return_video_type is None:
            return_video_type = self.return_video_type
            
        if return_video_type == "tensor":
            frames = torch.from_numpy(np.array(frames, dtype=np.float32))  # [T,H,W,C]
            frames = frames.permute(3, 0, 1, 2)[None,...] # [B,C,T,H,W]
            frames = frames / 127.5 - 1.0             # Normalize to [-1, 1]
            frames = frames.to(torch.bfloat16) # Default dtype: bf16
        elif return_video_type == "pil":
            frames = [Image.fromarray(frame) for frame in frames]  # List of PIL.Image
            if self.frame_processor is not None:
                frames = self.frame_processor(frames)
        elif return_video_type == "numpy":
            frames = np.array(frames, dtype=np.uint8)  # [T,H,W,C]
        elif return_video_type == "raw":
            frames = frames 
        else:
            raise ValueError(f"Unknown return_video_type: {self.return_video_type}")
        return frames

    def __getitem__(self, data_id):
        # 1. Get base data
        data = self.data[data_id % len(self.data)].copy()
        video_key = "video"
        
        # 2. Load main video
        full_video_path = self.rel_path_to_full(data[video_key], self.base_path)
        start_point = get_random_start_point(full_video_path) if self.random_start_point else 0
        video = self._load_video(full_video_path, start_point,frame_skip=1)
        
        # 3. Load action data (if needed)
        if self.load_action:
            action_state_latents = data["action_state_latents"]

            full_action_path = self.rel_path_to_full(action_state_latents, self.base_path)
            action_state_latents = LoadActionDataV3(action_max=self.action_max,
                                                    action_min=self.action_min,
                                                    num_frames=self.num_frames,
                                                    temporal_downsample=self.temporal_downsample,
                                                    action_type=self.action_type)(
                                                    file_path=full_action_path,
                                                    start_point=start_point
                                                )
            # pad action 
            num_agent = action_state_latents.shape[1]
            num_agent_to_pad = self.max_agent - num_agent
            if self.sep_action and num_agent > 1:
                # Randomly select one agent to keep, zero out others
                keep_idx = torch.randint(0, num_agent, (1,)).item()
                # Create mask: keep only selected agent
                mask = torch.zeros_like(action_state_latents)
                mask[:, keep_idx, :, :] = 1.0
                action_state_latents = action_state_latents * mask
                
            if num_agent_to_pad >0:
                pad_tensor = torch.zeros((1,num_agent_to_pad,action_state_latents.shape[2],action_state_latents.shape[3]),dtype=action_state_latents.dtype)
                action_state_latents = torch.cat([action_state_latents,pad_tensor],dim=1)
            data['action_state_latents'] = action_state_latents
            
            camera_path = data.get("camera_pose", None)
            if camera_path:
                camera_poses = self.load_camera_pose(camera_path)
                data["camera_pose"] = camera_poses

            data['action']={}
            data['action']['num_agents'] = num_agent
            data['action']['camera'] = data.get("camera_pose", None)
            data['action']['action'] = data.get("action_state_latents", None)
        
        # 4. Load environment observation data (if needed)
        if self.load_env_obv: 
            unique_key = self.get_unique_key(data) 
            
            # Get all video paths from the same episode
            env_obv_list = self.episode_to_data_ids.get(unique_key, [])
            env_obv_list = [os.path.join(self.base_path, vp) for vp in env_obv_list]
            env_obv_list = [vp for vp in env_obv_list if os.path.isfile(vp)]
            env_obv_list = sorted(env_obv_list)  # Ensure consistent ordering
            # Randomly sample the specified number of views
            target_num_views = self.num_views
            if len(env_obv_list) > target_num_views:
                env_obv_list = random.sample(env_obv_list, target_num_views)
            
            # Process data according to env observation type
            if self.env_obv_type == "memory":
                # Process memory mode
                env_videos = load_and_preprocess_videos(
                    env_obv_list,
                    mode="pad",
                    target_frames=1,
                    frame_stride=1,
                )  # (K, F, 3, H, W)
                
                env_videos = rearrange(env_videos, "K F C H W -> 1 F K C H W").contiguous()
                data["env_obv"] = env_videos # env_videos shape torch.Size([1, 1, 3, 3, 224, 224])
                # print(f"debug memory type: env_videos shape {env_videos.shape}")
            elif self.env_obv_type == "concat":
                # Process concat mode
                env_videos = []
                camera_pose_list = []
                
                for env_obv_path in env_obv_list:
                    # Load environment observation video
                    full_obv_path = self.rel_path_to_full(env_obv_path, self.base_path)
                    env_video_frames = self._load_video(full_obv_path, start_point,frame_skip=1)  # [1, C, F, H, W]
                    env_videos.append(env_video_frames)
                    
                    # Load corresponding camera poses
                    env_rel_path = os.path.relpath(env_obv_path, self.base_path)
                    camera_path = self.video_path_to_camera.get(env_rel_path, None)
                    if camera_path:
                        env_camera_poses = self.load_camera_pose(camera_path)
                        camera_pose_list.append(env_camera_poses)
                # b c f h w
                if self.return_video_type == "tensor":
                    env_videos = torch.cat(env_videos, dim=4).contiguous()  # (1, F, K*3, H, W)
                elif self.return_video_type == "pil":
                    # Each video is a list of PIL images; concat them into one list and process together
                    concat_videos = [ ]
                    for idx in range(self.num_frames):
                        frames_to_concat = [env_video[idx] for env_video in env_videos]  # idx-th frame of each video
                        # Concatenate PIL images on width
                        new_image = Image.new('RGB', (self.target_width * len(frames_to_concat), self.target_height))
                        for i, frame in enumerate(frames_to_concat):
                            new_image.paste(frame, (i * self.target_width, 0))
                        concat_videos.append(new_image)
                    env_videos = concat_videos
                video = env_videos 
                # Update data
                if camera_pose_list:
                    data['action']['camera'] = torch.stack(camera_pose_list, dim=1)
                data['env_obv'] = None
        else:
            data['env_obv'] = None
        data['video_path'] = os.path.relpath(full_video_path,self.base_path)
        data[video_key] = video
        return data
    def __len__(self):
        
        if self.load_from_cache:
            return min(self.max_entries, len(self.cached_data) * self.repeat)
        else:
            return min(self.max_entries, len(self.data) * self.repeat)

class RobotsVideoActionDataset(torch.utils.data.Dataset):
    def __init__(self, 
                video_dir: str,
                action_dir: str,
                video_params,
                return_video_type,
                random_start_point,
                # multiview config
                load_env_obv,
                env_obv_type, 
                num_views,
                num_camera_pose_frames,
                load_action,
                max_agent, 
                action_type=None,
                action_min=None,
                action_max=None,
                # dataset config 
                repeat=1,
                data_file_keys=tuple("video"),
                output_dir=None,
                max_entries=None, 
                temporal_downsample=4,
                sep_action=False,
                **kwargs,
    ):
        super(RobotsVideoActionDataset, self).__init__()
        
        self.video_dir = video_dir
        self.action_dir = action_dir
        self.max_entries = max_entries if max_entries is not None else 999999999
        self.random_start_point = random_start_point
        
        self.load_env_obv = load_env_obv
        self.env_obv_type = env_obv_type
        self.num_views = num_views
        
        self.load_action = load_action
        self.action_type = action_type
        self.max_agent = max_agent
        self.action_min = action_min
        self.action_max = action_max 
        self.num_camera_pose_frames = num_camera_pose_frames 
        self.temporal_downsample = temporal_downsample 
        
        self.video_params = video_params
        self.frame_skip = video_params.get('frame_skip', 1)
        self.target_height = video_params.get('height', 320)
        self.target_width = video_params.get('width', 640)
        self.num_frames = video_params.get('num_frames', 81)
        self.return_video_type = return_video_type 
        self.sep_action = sep_action
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.output_dir = output_dir 
        
        # Build file list from video_dir
        self._build_file_list()
        self.frame_processor = None 
        
    def _build_file_list(self):
        """Build list of (video_path, action_path) pairs from directories."""
        video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])
        
        self.data = []
        for video_file in video_files:
            video_path = os.path.join(self.video_dir, video_file)
            action_file = video_file.replace('.mp4', '.npy')
            action_path = os.path.join(self.action_dir, action_file)
            
            if os.path.exists(action_path):
                self.data.append({
                    'video': video_path,
                    'action': action_path,
                    'video_filename': video_file
                })
            else:
                print(f"Warning: No matching action file for {video_file}")
        
        print(f"RobotsVideoActionDataset initialized with {len(self.data)} samples")
    
    def _load_raw_video(self, path: str, start_point: int, frame_skip: int) -> list:
        """
        Load raw video frames as Numpy array with shape [T, H, W, C] and dtype uint8.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_point)
        if self.num_frames is None: 
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames_to_read = total_frames - start_point
        else:
            num_frames_to_read = self.num_frames
            
        for i in range(num_frames_to_read):
            ret, frame = cap.read()
            if not ret:
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                raise ValueError(f"Path {path} Failed at frame {i}, pos {start_point + i*frame_skip}, actual {actual_pos}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize on demand
            if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            frames.append(frame)  
            
            # Skip frame_skip-1 frames
            if frame_skip > 1 and i < self.num_frames - 1:
                for _ in range(frame_skip - 1):
                    if not cap.grab():
                        raise ValueError(f"Grab failed at skip {i}")
        cap.release()
        return frames

    def _load_video(self, path: str, start: int, frame_skip: int, return_video_type=None):
        """Load video frames and return in the specified format."""
        frames = self._load_raw_video(path, start, frame_skip)
        if return_video_type is None:
            return_video_type = self.return_video_type
            
        if return_video_type == "tensor":
            frames = torch.from_numpy(np.array(frames, dtype=np.float32))
            frames = frames.permute(3, 0, 1, 2)[None, ...]  # [B, C, T, H, W]
            frames = frames / 127.5 - 1.0  # Normalize to [-1, 1]
            frames = frames.to(torch.bfloat16)
        elif return_video_type == "pil":
            frames = [Image.fromarray(frame) for frame in frames]
            if self.frame_processor is not None:
                frames = self.frame_processor(frames)
        elif return_video_type == "numpy":
            frames = np.array(frames, dtype=np.uint8)  # [T, H, W, C]
        elif return_video_type == "raw":
            frames = frames 
        else:
            raise ValueError(f"Unknown return_video_type: {self.return_video_type}")
        return frames

    def __getitem__(self, idx):
        # Handle repeat
        data_id = idx % len(self.data)
        data_entry = self.data[data_id]
        
        # Load video
        video_path = data_entry['video']
        start_point = get_random_start_point(video_path) if self.random_start_point else 0
        video = self._load_video(video_path, start_point, frame_skip=1)
        
        result = {
            'video': video,
            'video_filename': data_entry['video_filename']
        }
        
        # Load action (same logic as original)
        if self.load_action:
            action_path = data_entry['action']
            action_state_latents = np.load(action_path)[0]
            action_state_latents = torch.from_numpy(action_state_latents).float()[::4]

            num_agent = action_state_latents.shape[1]
            num_agent_to_pad = self.max_agent - num_agent
            
            if self.sep_action and num_agent > 1:
                # Randomly select one agent to keep, zero out others
                keep_idx = torch.randint(0, num_agent, (1,)).item()
                mask = torch.zeros_like(action_state_latents)
                mask[:, keep_idx, :, :] = 1.0
                action_state_latents = action_state_latents * mask
                
            if num_agent_to_pad > 0:
                pad_tensor = torch.zeros(
                    (1, num_agent_to_pad, action_state_latents.shape[2], action_state_latents.shape[3]),
                    dtype=action_state_latents.dtype,
                    device=action_state_latents.device
                )
                action_state_latents = torch.cat([action_state_latents, pad_tensor], dim=1)
            
            result['action_state_latents'] = action_state_latents
            result['action'] = {
                'num_agents': num_agent,
                'action': action_state_latents
            }
        
        # Note: load_env_obv not supported in this simplified version
        # (would need additional metadata to group related videos)
        result['env_obv'] = None
        
        return result
    
    def __len__(self):
        return min(self.max_entries, len(self.data) * self.repeat)
    