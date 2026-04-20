
import os
import cv2
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

# -------------------- Caching and optimization utilities --------------------
# Cache frequently accessed data to reduce redundant computation
_text_size_cache = {}

def get_text_size(text, font, scale, thick):
    """Cache text size calculations to avoid repeated cv2.getTextSize calls."""
    cache_key = (text, font, scale, thick)
    if cache_key not in _text_size_cache:
        _text_size_cache[cache_key] = cv2.getTextSize(text, font, scale, thick)[0]
    return _text_size_cache[cache_key]

def scale_for_height(target_h, font=cv2.FONT_HERSHEY_SIMPLEX, thick=1):
    """
    Return the scale that makes letter height approximately target_h.
    Measured letter height is approximately 18.6 * scale (when thick=1).
    """
    base_h = 18.6          # Empirical constant
    return target_h / base_h

def make_overlay(shape, action_row, key_order,video_w: int, video_h: int, 
                               font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0, thick=2, 
                               color_on=(0, 0, 255, 200), color_off=(200, 200, 200, 100), 
                               amp_mouse_threshold=0.01):

    h, w = shape[:2] # (320,640) - (480-960) - (320-320)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    if key_order is None:
        key_order = list(action_row.keys())
    state = {k: action_row[k] for k in key_order if k in action_row}
    # Reference dimensions
    cell = int(w / 16)
    gap = int(h / 30 ) # 480 / 30 = 6 ; 
    margin = int(h / 48) # 480 / 48 = 10 
    # Compute scale based on height
    
    scale = scale_for_height(cell//2 , font, thick)
    # Predefine joystick parameters
    pad_r = h // 8 
    offset = w // 2
    cx_l = margin + pad_r + offset
    cy_l = h //2  - pad_r - margin
    cx_r = w - margin - pad_r
    cy_r = h //2 - pad_r - margin

    # -------------------- 1) WASD + Q/E/F + Shift/Ctrl + Space --------------------
    # Left half layout, all keys do not exceed the midline
    total_w = w // 2
    q_w = e_w = cell
    w_w = cell
    shift_w = int(cell * 1.6)
    ctrl_w = shift_w
    space_w = int(cell * 1.8)
    mouse_key_w = int(cell * 1.2)

    # Recompute layout: QWE directly above ASD, Space/L/R do not exceed midline
    # Base row: A-S-D
    base_block_w = 3 * cell + 2 * gap  # Width of A-S-D
    # Upper row: Q-W-E, W directly above S
    upper_block_w = q_w + w_w + e_w + 2 * gap  # Width of Q-W-E
    
    # Use the larger of the two as the centering baseline
    block_w = max(base_block_w, upper_block_w)
    x_start = (total_w - block_w) // 2
    
    # Recompute horizontal positions - QWE directly above ASD
    # ASD row
    x_a = x_start + (block_w - base_block_w) // 2
    x_s = x_a + cell + gap
    x_d = x_s + cell + gap
    
    # QWE row - W directly above S, Q and E directly above A and D respectively
    x_q = x_a - (q_w - cell) // 2  # Q directly above A
    x_w = x_s - (w_w - cell) // 2  # W directly above S
    x_e = x_d - (e_w - cell) // 2  # E directly above D
    x_f = x_e + e_w + gap  # F to the right of E
    
    # Space - in the left half, not exceeding midline
    x_space =  x_d + cell + gap  # Space aligned to the right in the left half
    
    # Ctrl and Shift - to the left of Space
    x_shift = x_a - shift_w - gap
    x_ctrl = x_space 
    
    # Mouse buttons - ensure they are in the left half, not exceeding midline
    x_mouse_l = x_space 
    x_mouse_r = x_mouse_l + mouse_key_w + gap

    # Vertical anchors
    y_base = h - margin - cell          # Bottom row (a-s-d-shift-space)
    y_w = y_base - cell - gap           # Middle row (q-w-e)
    y_ctrl = y_w - cell - gap           # Top row (ctrl)
    y_mouse = y_ctrl - cell - gap       # Topmost row (mouse L/R)

    def draw_key_with_border(text, x, y, width, height, is_pressed):
        """Draw a key with a border."""
        color = color_on if is_pressed else color_off
        # Draw border
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
        # Draw text (centered) - use cached size for optimization
        text_size = get_text_size(text, font, scale, thick)
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        cv2.putText(overlay, text, (text_x, text_y), font, scale, color, thick, cv2.LINE_AA)

    # Draw all keys
    draw_key_with_border('CTRL', x_ctrl, y_ctrl, ctrl_w, cell, state.get('ctrl', 0))
    draw_key_with_border('SHIFT', x_shift, y_base, shift_w, cell, state.get('shift', 0))
    
    draw_key_with_border('Q', x_q, y_w, q_w, cell, state.get('q', 0))
    draw_key_with_border('W', x_w, y_w, w_w, cell, state.get('w', 0))  # W directly above S
    draw_key_with_border('E', x_e, y_w, e_w, cell, state.get('e', 0))
    draw_key_with_border('F', x_f, y_w, e_w, cell, state.get('f', 0))
    
    draw_key_with_border('A', x_a, y_base, cell, cell, state.get('a', 0))
    draw_key_with_border('S', x_s, y_base, cell, cell, state.get('s', 0))
    draw_key_with_border('D', x_d, y_base, cell, cell, state.get('d', 0))
    draw_key_with_border('SPACE', x_space, y_base, space_w, cell, state.get('space', 0))
    
    # draw_key_with_border('L', x_mouse_l, y_mouse, mouse_key_w, cell, state.get('mouse_left', 0))
    # draw_key_with_border('R', x_mouse_r, y_mouse, mouse_key_w, cell, state.get('mouse_right', 0))

    # -------------------- 2) Pad buttons (0-9, excluding 6) --------------------
    btn_lst = [f'button_{i}' for i in [0,1,2,3,4,5,7,8,9]]
    cols = 3
    total_btn = len(btn_lst)
    rows = (total_btn + cols - 1) // cols
    
    btn_w = int(cell * 1.2)
    btn_h = cell
    gap_x, gap_y = 8, 6
    
    total_block_w = cols * btn_w + (cols - 1) * gap_x
    total_block_h = rows * btn_h + (rows - 1) * gap_y
    
    # Position: ensure buttons do not exceed the bottom of the screen
    # First check if there is enough space at the bottom
    bottom_margin = margin + cell + gap
    max_y0 = h - bottom_margin - total_block_h
    
    # Horizontally centered between the two joysticks
    left_edge = cx_l + pad_r + gap
    right_edge = cx_r - pad_r - gap
    x0 = (left_edge + right_edge - total_block_w) // 2
    
    # Vertical position: prefer center of joysticks, but shift up if it would exceed the bottom
    ideal_y0 = h - pad_r - margin - total_block_h // 2
    y0 = min(ideal_y0, max_y0)
    
    # Ensure y0 is not less than the top margin
    y0 = max(y0, margin + cell)
    
    for idx, k in enumerate(btn_lst):
        r, c = divmod(idx, cols)
        x = x0 + c * (btn_w + gap_x)
        y = y0 + r * (btn_h + gap_y)
        label = k.split('_')[1]
        is_pressed = state.get(k, 0)
        color = color_on if is_pressed else color_off
        
        # Draw border
        cv2.rectangle(overlay, (x, y), (x + btn_w, y + btn_h), color, 2)
        # Draw text (centered) - use cached size for optimization
        text_size = get_text_size(label, font, scale, thick)
        text_x = x + (btn_w - text_size[0]) // 2
        text_y = y + (btn_h + text_size[1]) // 2
        cv2.putText(overlay, label, (text_x, text_y), font, scale, color, thick, cv2.LINE_AA)

    # -------------------- 3) Joystick discs --------------------
    # Outer rings
    for cx, cy in [(cx_l, cy_l), (cx_r, cy_r)]:
        cv2.circle(overlay, (cx, cy), pad_r, (255, 255, 255, 100), 2)

    def draw_stick(cx, cy, dx, dy, label):
        """Draw a stick: idle -> small dot; moving -> arrow"""
        amp = math.hypot(dx, dy)
        if amp < 0.1:
            # Within dead zone, draw a small dot. Must be consistent with data collection logic.
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 0, 200), -1)
        else:
            scale_len = min(amp, 1.0) * pad_r
            end_x = int(cx + dx * scale_len)
            end_y = int(cy + dy * scale_len)
            cv2.arrowedLine(overlay, (cx, cy), (end_x, end_y), (0, 255, 0, 200), thickness=3, tipLength=0.2)
        cv2.putText(overlay, label, (cx - 10, cy - pad_r - 5), font, 0.6, (255, 255, 255, 150), 1)

    lx,ly,rx,ry = action_row.get('axis_0', 0.0), action_row.get('axis_1', 0.0), action_row.get('axis_2', 0.0), action_row.get('axis_3', 0.0)
    draw_stick(cx_l, cy_l, lx, ly, 'Move')
    draw_stick(cx_r, cy_r, rx, ry, 'Scope')

    # -------------------- 4) Mouse direction disc --------------------
    dx = float(action_row.get('norm_dx', 0.0))
    dy = float(action_row.get('norm_dy', 0.0))
    cx_mouse = margin + pad_r
    cy_mouse = h // 2
    cv2.circle(overlay, (cx_mouse, cy_mouse), pad_r, (255, 255, 255, 100), 2)
    amp_mouse = math.hypot(dx, dy)
    if  amp_mouse < amp_mouse_threshold:
        cv2.circle(overlay, (cx_mouse, cy_mouse), 4, (255, 0, 255, 200), -1)
    else:
        end_x = int(cx_mouse + dx * pad_r) 
        end_y = int(cy_mouse + dy * pad_r) 
        cv2.arrowedLine(overlay, (cx_mouse, cy_mouse), (end_x, end_y), (255, 0, 255, 200), thickness=3, tipLength=0.2)
    # Shift "M" 10 pixels to the left (was centered, now left-aligned)
    m_x = cx_mouse - pad_r - 10
    m_y = cy_mouse - pad_r - 5
    cv2.putText(overlay, 'M', (m_x, m_y),
                font, 0.6, (255, 255, 255, 150), 1)

    # Write dx, dy to the right of "M" on the same line (two decimal places)
    text = f"{dx:+.2f},{dy:+.2f}"
    cv2.putText(overlay, text,(m_x + 15, m_y),          # Immediately to the right of M
                font, 0.4, (200, 200, 200, 180), 1)
    
    # -------------------- 5) Mouse actual position red dot --------------------
    # px = int(action_row.get('mouse_x', 0.0) * w / video_w)
    # py = int(action_row.get('mouse_y', 0.0) * h / video_h)
    
    # cv2.circle(overlay, (px, py), 6, (0, 0, 255, 255), -1)

    return overlay


# Functions to visualize actions on video frames
import cv2
import subprocess
import numpy as np 
import PIL 

def visualize_action_on_frames(frames,df, key_order, dst_fps=60, dst_w=640,dst_h=320,use_overlay=True):
    # Convert frames: PIL.Image to np.ndarray (bgr)
    bgr_frames = []
    for frame in frames:
        frame_np = np.array(frame.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        bgr_frames.append(frame_bgr)
    # Call visualize_action_on_video
    return_frames = []    
    for frame, action_row in zip(bgr_frames, df.to_dict('records')):
        overlay = make_overlay(
            shape = frame.shape,
            key_order = key_order,
            action_row = action_row,
            video_h=dst_h,
            video_w=dst_w,
            amp_mouse_threshold=0.0
        )
        alpha = overlay[:, :, 3:4] / 255.0
        overlay_rgb = overlay[:, :, :3]
        frame_float = frame.astype(np.float32)
        overlay_float = overlay_rgb.astype(np.float32)
        
        # Blend only where alpha > 0
        if use_overlay:
            mask = alpha[:, :, 0] > 0
        else:
            mask = np.zeros_like(alpha[:,:,0], dtype=bool)
            
        frame_float[mask] = (overlay_float[mask] * alpha[mask] + 
                        frame_float[mask] * (1 - alpha[mask]))
        
        frame = frame_float.astype(np.uint8)
        # Convert to PIL.Image 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        return_frames.append(frame)
    return return_frames 

def visualize_action_on_video(frames, df, output_video,key_order, dst_fps=60, dst_w=640,dst_h=320,use_overlay=True):
    # Video is loaded as a list of frames or array of frames 
    # Frames: original BGR frames [0,255]
    # Action: df of actions
    tmp_avi = output_video + ".avi" 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(tmp_avi, fourcc, dst_fps , (dst_w, dst_h))
    return_frames = []
    for frame, action_row in zip(frames, df.to_dict('records')):

        overlay = make_overlay(
            shape = frame.shape,
            key_order = key_order,
            action_row = action_row,
            video_h=dst_h,
            video_w=dst_w,
            amp_mouse_threshold=0.0
        )
        alpha = overlay[:, :, 3:4] / 255.0
        overlay_rgb = overlay[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_float = frame.astype(np.float32)
        overlay_float = overlay_rgb.astype(np.float32)
        
        # Blend only where alpha > 0
        if use_overlay:
            mask = alpha[:, :, 0] > 0
        else:
            mask = np.zeros_like(alpha[:,:,0], dtype=bool)
        frame_float[mask] = (overlay_float[mask] * alpha[mask] + 
                        frame_float[mask] * (1 - alpha[mask]))
        
        frame = frame_float.astype(np.uint8)
        return_frames.append(frame)
        writer.write(frame)

    writer.release()
    
    subprocess.run(['ffmpeg', '-y', '-i', tmp_avi, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast','-r', str(dst_fps),
                    '-movflags', '+faststart', '-loglevel', 'error', output_video], check=True)
    
    os.remove(tmp_avi)
    return return_frames
     
