#### abstract video demo

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float

def create_abstract_video(
    input_video_path: Path,
    output_video_path: Path,
    frames_per_minute: int = 60,
    duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    frame_difference_threshold: float = 1.0,
    target_duration: float = 1.5
) -> None:
    """Create an abstract video with specified target duration."""
    def _calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        if frame1 is None or frame2 is None:
            return 0.0
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(diff))

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    if duration:
        video_duration = min(duration, video_duration)
        total_frames = int(min(total_frames, duration * fps))
    
    target_frames = max(1, min(
        int((video_duration / 60) * frames_per_minute),
        total_frames,
        max_frames if max_frames is not None else float('inf')
    ))
    
    sample_interval = max(1, total_frames // (target_frames * 2))
    frame_candidates = []
    prev_frame = None
    frame_count = 0
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            score = _calculate_frame_difference(frame, prev_frame)
            if score > frame_difference_threshold:
                timestamp = frame_count / fps
                frame_candidates.append((frame_count, frame, score, timestamp))
            prev_frame = frame.copy()
            
        frame_count += 1
        
    cap.release()
    frame_candidates.sort(key=lambda x: x[0])
    
    if len(frame_candidates) > target_frames:
        step = len(frame_candidates) / target_frames
        indices = [int(i * step) for i in range(target_frames)]
        selected_frames = [frame_candidates[i] for i in indices]
    else:
        selected_frames = frame_candidates

    if not selected_frames:
        raise ValueError("No keyframes were selected")
    
    frame_height, frame_width = selected_frames[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Calculate required repeats for target duration
    total_frames_needed = int(target_duration * fps)
    frames_count = len(selected_frames)
    base_repeats = total_frames_needed // frames_count
    remainder = total_frames_needed % frames_count
    frame_repeats = [base_repeats + 1 if i < remainder else base_repeats 
                    for i in range(frames_count)]
    
    output_fps = sum(frame_repeats) / target_duration
    out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (frame_width, frame_height))
    
    for (frame_num, frame, score, timestamp), repeats in zip(selected_frames, frame_repeats):
        for _ in range(repeats):
            out.write(frame)
    
    out.release()
    logger.info(f"Created {target_duration}s abstract video: {output_video_path.name}")

def process_directory(input_dir: str):
    """Process all MP4 files in directory with target duration 1.5s and 60fpm."""
    input_path = Path(input_dir)
    output_dir = input_path.parent / f"{input_path.name}_1_5_seconds"
    output_dir.mkdir(exist_ok=True)
    
    mp4_files = list(input_path.glob("*.mp4"))
    if not mp4_files:
        logger.warning(f"No MP4 files found in {input_path}")
        return
    
    logger.info(f"Processing {len(mp4_files)} MP4 files to {output_dir}")
    
    for input_file in mp4_files:
        output_file = output_dir / input_file.name
        try:
            create_abstract_video(
                input_file,
                output_file,
                frames_per_minute=60,
                target_duration=1.5
            )
        except Exception as e:
            logger.error(f"Failed to process {input_file.name}: {str(e)}")
    
    logger.info(f"Completed processing {len(mp4_files)} files")

if __name__ == "__main__":
    input_directory = r"C:\Users\DELL\Downloads\Genshin_StarRail_Longshu_Sketch_Tail_Videos_Reversed"
    process_directory(input_directory)
```

#### use this to get depth image
#### https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
#### https://huggingface.co/spaces/svjack/Depth-Anything-V2

```python
#### 规则可能就能够生成

import cv2
import numpy as np
from PIL import Image
import io

# Function to convert image to sketch with adjustable outline thickness
def image_to_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = 255 - gray_image
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blurred = 255 - blurred_image
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    return sketch

def generate_transition_video(image_path, output_path, duration_sec=5, frame_rate=30):
    """
    生成从素描渐变到原图的视频

    参数：
    image_path : str - 输入图片路径
    output_path : str - 输出视频路径
    duration_sec : int - 视频总时长（秒）
    frame_rate : int - 视频帧率（默认30fps）
    """
    # 读取并预处理原图
    original = cv2.imread(image_path)
    h, w = original.shape[:2]

    # 生成素描图（调整为三通道）
    sketch = image_to_sketch(original)
    #print(sketch)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)  # 转换通道格式[5](@ref)

    # 初始化视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 推荐兼容性较好的编码器[2](@ref)
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))

    # 计算过渡参数
    total_frames = int(duration_sec * frame_rate)
    alpha_step = 1.0 / total_frames  # 混合权重步长

    # 生成渐变帧序列
    for i in range(total_frames):
        alpha = 1.0 - i * alpha_step
        beta = 1.0 - alpha

        # 图像混合（支持任意通道数的图像混合）
        blended = cv2.addWeighted(sketch_bgr, alpha, original, beta, 0)

        video.write(blended)

    # 清理资源
    video.release()
    cv2.destroyAllWindows()

generate_transition_video("化物语封面.jpeg", "化物语渐变.mp4", 5, 30)

generate_transition_video("竹林万叶.jpg", "竹林万叶渐变.mp4", 5, 30)

generate_transition_video("bloom_tree.png", "bloom_tree渐变.mp4", 5, 30)


#### https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

#### 根据深度 进行顺序渐变

import cv2
import numpy as np
from typing import Literal

def image_to_sketch(image: np.ndarray) -> np.ndarray:
    """Convert image to pencil sketch"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = 255 - blurred
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return sketch
    #return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)  # Return 3-channel image

def generate_transition_video(
    image_path: str,
    output_path: str,
    depth_map_path: str = None,
    render_order: Literal['far_to_near', 'near_to_far'] = 'far_to_near',
    duration_sec: float = 5.0,
    frame_rate: int = 30,
    depth_blur: int = 15,
    num_layers: int = 10,
    debug_visualize: bool = False
) -> None:
    """
    Generate transition video using depth-aware layered rendering

    Args:
        num_layers: Number of depth layers to split
        ...其他参数同上...
    """
    # ===== 1. 加载和预处理 =====
    original = cv2.imread(image_path)
    sketch = image_to_sketch(original)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    h, w = original.shape[:2]

    # ===== 2. 深度分层处理 =====
    depth_layers = []
    if depth_map_path:
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        depth_map = cv2.resize(depth_map, (w, h))
        depth_map = cv2.GaussianBlur(depth_map, (depth_blur, depth_blur), 0)
        depth_map = depth_map.astype(np.float32) / 255.0

        if render_order == 'near_to_far':
            depth_map = 1.0 - depth_map

        # 创建深度分层蒙版
        layer_bins = np.linspace(0, 1, num_layers + 1)
        for i in range(num_layers):
            mask = np.logical_and(depth_map >= layer_bins[i],
                                depth_map <= layer_bins[i+1])
            depth_layers.append(mask.astype(np.float32))
    else:
        # 无深度图时全图作为单层
        depth_layers = [np.ones((h, w), np.float32)]

    # ===== 3. 生成过渡动画 =====
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))
    total_frames = int(duration_sec * frame_rate)
    layer_duration = duration_sec / num_layers

    for frame_idx in range(total_frames):
        current_time = frame_idx / frame_rate
        blended = original.copy().astype(np.float32)

        for layer_idx, layer_mask in enumerate(depth_layers):
            # 计算当前层的进度
            layer_start = layer_idx * layer_duration
            layer_progress = np.clip((current_time - layer_start) / layer_duration, 0, 1)

            # 生成混合蒙版
            layer_alpha = layer_mask * (1 - layer_progress)
            layer_alpha = np.repeat(layer_alpha[..., np.newaxis], 3, axis=2)

            # 分层混合
            blended = blended * (1 - layer_alpha) + sketch_bgr.astype(np.float32) * layer_alpha

        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if debug_visualize:
            cv2.imshow('Blended', blended)
            if cv2.waitKey(1) == 27:
                break

        video.write(blended)

    video.release()
    if debug_visualize:
        cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

generate_transition_video(
        image_path="化物语封面.jpeg",
        output_path="化物语封面深度渐变.avi",
        depth_map_path="化物语封面深度.png",  # Optional
        render_order='far_to_near',     # or 'near_to_far'
        duration_sec=3.0,
        frame_rate=30,
        depth_blur=15,
        debug_visualize=False
    )

generate_transition_video(
        image_path="bloom_tree.png",
        output_path="bloom_tree深度渐变.avi",
        depth_map_path="bloom_tree_depth.png",  # Optional
        render_order='near_to_far',     # or 'near_to_far'
        duration_sec=3.0,
        frame_rate=30,
        depth_blur=15,
        debug_visualize=False
    )

```

#### Integrate with Cartoon Segmentation technology
#### have 
#### https://huggingface.co/spaces/svjack/AnimeIns_Depth_Sketch_Video_CPU
#### sketch_video_app.py
#### sketch_video_app_from_blank.py

```python
#### https://huggingface.co/spaces/svjack/Depth-Anything-V2
#### app.py

import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
client = Client("http://localhost:7861/")

# Define paths
source_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
output_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all files in source folder
for filename in tqdm(os.listdir(source_folder)):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', ".webp")):
        continue

    # Process the image
    file_path = os.path.join(source_folder, filename)
    try:
        result = client.predict(
            image=handle_file(file_path),
            api_name="/on_submit"
        )

        # Copy the depth image to output folder with same filename
        depth_image_path = result[1]
        output_path = os.path.join(output_folder, filename)
        copy2(depth_image_path, output_path)

        print(f"Processed {filename} successfully")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print("All images processed!")

#### https://huggingface.co/spaces/svjack/AnimeIns_Depth_Sketch_Video_CPU
#### sketch_video_app.py

import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
client = Client("http://localhost:7860")

# Define paths
sketch_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
depth_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"
output_folder = "Genshin_StarRail_Longshu_Sketch_Guide_to_Color_Videos_character_first"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of files in both folders (only process files that exist in both)
sketch_files = set(os.listdir(sketch_folder))
depth_files = set(os.listdir(depth_folder))
common_files = sketch_files.intersection(depth_files)

# Process each matching file pair
for filename in tqdm(common_files):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', ".webp" )):
        continue

    try:
        # Prepare file paths
        sketch_path = os.path.join(sketch_folder, filename)
        depth_path = os.path.join(depth_folder, filename)

        # Process the image pair
        result = client.predict(
            original_image=handle_file(sketch_path),
            depth_map=handle_file(depth_path),
            render_order="character_first",
            duration=3,
            api_name="/process_images"
        )

        # Generate output video path (replace image extension with .mp4)
        video_name = os.path.splitext(filename)[0] + ".mp4"
        output_path = os.path.join(output_folder, video_name)

        # Copy the generated video to output folder
        copy2(result["video"], output_path)

        print(f"Processed {filename} successfully. Video saved as {video_name}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print(f"Processing complete! Videos saved in {output_folder}")

import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
client = Client("http://localhost:7860")

# Define paths
sketch_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
depth_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"
output_folder = "Genshin_StarRail_Longshu_Sketch_Guide_to_Color_Videos_far_to_near"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of files in both folders (only process files that exist in both)
sketch_files = set(os.listdir(sketch_folder))
depth_files = set(os.listdir(depth_folder))
common_files = sketch_files.intersection(depth_files)

# Process each matching file pair
for filename in tqdm(common_files):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif' , ".webp" )):
        continue

    try:
        # Prepare file paths
        sketch_path = os.path.join(sketch_folder, filename)
        depth_path = os.path.join(depth_folder, filename)

        # Process the image pair
        result = client.predict(
            original_image=handle_file(sketch_path),
            depth_map=handle_file(depth_path),
            render_order="far_to_near",
            duration=3,
            api_name="/process_images"
        )

        # Generate output video path (replace image extension with .mp4)
        video_name = os.path.splitext(filename)[0] + ".mp4"
        output_path = os.path.join(output_folder, video_name)

        # Copy the generated video to output folder
        copy2(result["video"], output_path)

        print(f"Processed {filename} successfully. Video saved as {video_name}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print(f"Processing complete! Videos saved in {output_folder}")

import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
client = Client("http://localhost:7860")

# Define paths
sketch_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
depth_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"
output_folder = "Genshin_StarRail_Longshu_Sketch_Guide_to_Color_Videos_near_to_far"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of files in both folders (only process files that exist in both)
sketch_files = set(os.listdir(sketch_folder))
depth_files = set(os.listdir(depth_folder))
common_files = sketch_files.intersection(depth_files)

# Process each matching file pair
for filename in tqdm(common_files):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif' , ".webp" )):
        continue

    try:
        # Prepare file paths
        sketch_path = os.path.join(sketch_folder, filename)
        depth_path = os.path.join(depth_folder, filename)

        # Process the image pair
        result = client.predict(
            original_image=handle_file(sketch_path),
            depth_map=handle_file(depth_path),
            render_order="near_to_far",
            duration=3,
            api_name="/process_images"
        )

        # Generate output video path (replace image extension with .mp4)
        video_name = os.path.splitext(filename)[0] + ".mp4"
        output_path = os.path.join(output_folder, video_name)

        # Copy the generated video to output folder
        copy2(result["video"], output_path)

        print(f"Processed {filename} successfully. Video saved as {video_name}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print(f"Processing complete! Videos saved in {output_folder}")
```

#### sketch_video_app_from_blank.py
```python
import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
client = Client("http://localhost:7860/")

# Define paths
sketch_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
depth_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"
output_base_folder = "Genshin_StarRail_Longshu_Blank_Sketch_Color_Videos"

sketch_folder = "ACG_Cover_Images"
depth_folder = "ACG_Cover_Depth_Images"
output_base_folder = "ACG_Cover_Blank_Sketch_Color_Videos"

# Create base output folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

# Get list of files in both folders (only process files that exist in both)
sketch_files = set(os.listdir(sketch_folder))
depth_files = set(os.listdir(depth_folder))
common_files = sketch_files.intersection(depth_files)

# Define all possible transition combinations
transitions = ["character_first", "near_to_far", "far_to_near"]
combinations = [(first, second) for first in transitions for second in transitions]

# Process each matching file pair with all transition combinations
for filename in tqdm(common_files):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', ".webp")):
        continue

    # Prepare file paths
    sketch_path = os.path.join(sketch_folder, filename)
    depth_path = os.path.join(depth_folder, filename)

    for first_transition, second_transition in combinations:
        try:
            # Create subfolder for this transition combination if it doesn't exist
            combo_folder = f"{first_transition}_{second_transition}"
            #output_folder = os.path.join(output_base_folder, combo_folder)
            #os.makedirs(output_folder, exist_ok=True)

            # Process the image pair with current transition combination
            result = client.predict(
                original_image=handle_file(sketch_path),
                depth_map=handle_file(depth_path),
                first_transition=first_transition,
                second_transition=second_transition,
                duration=6,
                api_name="/process_images"
            )

            #print(result)
            
            # Generate output video path with transition info in filename
            base_name = os.path.splitext(filename)[0]
            video_name = f"{base_name}_{first_transition}_{second_transition}.mp4"
            output_path = os.path.join(output_base_folder, video_name)

            # Copy the generated video to output folder
            copy2(result["video"], output_path)

            print(f"Processed {filename} with {first_transition}/{second_transition} successfully. Video saved as {video_name}")

        except Exception as e:
            print(f"Error processing {filename} with {first_transition}/{second_transition}: {str(e)}")

print(f"Processing complete! Videos saved in {output_base_folder}")

```

```python
import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
client = Client("http://localhost:7861/")

# Define paths
sketch_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
depth_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"
output_base_folder = "Genshin_StarRail_Longshu_Blank_Sketch_Color_Videos"

sketch_folder = "ACG_Cover_Images"
depth_folder = "ACG_Cover_Depth_Images"
output_base_folder = "ACG_Cover_Blank_Sketch_Color_Videos"

sketch_folder = "genshin_manga_images"
depth_folder = "genshin_manga_depth_images"
output_base_folder = "Genshin_Manga_Blank_Sketch_Color_Videos"

# Create base output folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

# Get list of files in both folders (only process files that exist in both)
sketch_files = set(os.listdir(sketch_folder))
depth_files = set(os.listdir(depth_folder))
common_files = sketch_files.intersection(depth_files)

# Define all possible transition combinations
#transitions = ["character_first", "near_to_far", "far_to_near"]
transitions = ["character_first"]
combinations = [(first, second) for first in transitions for second in transitions]

# Process each matching file pair with all transition combinations
for filename in tqdm(common_files):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', ".webp")):
        continue

    # Prepare file paths
    sketch_path = os.path.join(sketch_folder, filename)
    depth_path = os.path.join(depth_folder, filename)

    for first_transition, second_transition in combinations:
        try:
            # Create subfolder for this transition combination if it doesn't exist
            combo_folder = f"{first_transition}_{second_transition}"
            #output_folder = os.path.join(output_base_folder, combo_folder)
            #os.makedirs(output_folder, exist_ok=True)

            # Process the image pair with current transition combination
            result = client.predict(
                original_image=handle_file(sketch_path),
                depth_map=handle_file(depth_path),
                first_transition=first_transition,
                second_transition=second_transition,
                duration=6,
                api_name="/process_images"
            )

            #print(result)
            
            # Generate output video path with transition info in filename
            base_name = os.path.splitext(filename)[0]
            #video_name = f"{base_name}_{first_transition}_{second_transition}.mp4"
            video_name = f"{base_name}.mp4"
            output_path = os.path.join(output_base_folder, video_name)

            # Copy the generated video to output folder
            copy2(result["video"], output_path)

            print(f"Processed {filename} with {first_transition}/{second_transition} successfully. Video saved as {video_name}")

        except Exception as e:
            print(f"Error processing {filename} with {first_transition}/{second_transition}: {str(e)}")

print(f"Processing complete! Videos saved in {output_base_folder}")

```

#### sketch_video_app_from_blank_direction.py
```python
import os
from gradio_client import Client, handle_file
from shutil import copy2
from tqdm import tqdm

# Initialize client
#client = Client("https://2b705df0809ccdc76b.gradio.live/")
client = Client("http://localhost:7860/")

# Define paths
sketch_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Images"
depth_folder = "Genshin_StarRail_Longshu_Sketch_Guide_Depth_Images"
output_base_folder = "Genshin_StarRail_Longshu_Blank_Sketch_Color_Direction_Videos"

sketch_folder = "ACG_Cover_Images"
depth_folder = "ACG_Cover_Depth_Images"
output_base_folder = "ACG_Cover_Blank_Sketch_Color_Direction_Videos"

# Create base output folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

# Get list of files in both folders (only process files that exist in both)
sketch_files = set(os.listdir(sketch_folder))
depth_files = set(os.listdir(depth_folder))
common_files = sketch_files.intersection(depth_files)

# Define all possible transition and direction combinations
transitions = ["character_first", "near_to_far", "far_to_near"]
directions = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]

# Generate all combinations of transitions and directions
combinations = [
    (first_trans, second_trans, first_dir, second_dir)
    for first_trans in transitions
    for second_trans in transitions
    for first_dir in directions
    for second_dir in directions
]

# Process each matching file pair with all transition/direction combinations
for filename in tqdm(common_files):
    # Skip non-image files (optional)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', ".webp")):
        continue

    # Prepare file paths
    sketch_path = os.path.join(sketch_folder, filename)
    depth_path = os.path.join(depth_folder, filename)

    for first_transition, second_transition, first_direction, second_direction in combinations:
        try:
            # Process the image pair with current combination
            result = client.predict(
                original_image=handle_file(sketch_path),
                depth_map=handle_file(depth_path),
                first_transition=first_transition,
                second_transition=second_transition,
                first_direction=first_direction,
                second_direction=second_direction,
                duration=6,
                api_name="/process_images"
            )

            # Generate output video path with transition/direction info in filename
            base_name = os.path.splitext(filename)[0]
            video_name = f"{base_name}_{first_transition}_{second_transition}_{first_direction}_{second_direction}.mp4"
            output_path = os.path.join(output_base_folder, video_name)

            # Copy the generated video to output folder
            copy2(result["video"], output_path)

            print(f"Processed {filename} with {first_transition}/{second_transition} and {first_direction}/{second_direction} successfully. Video saved as {video_name}")

        except Exception as e:
            print(f"Error processing {filename} with {first_transition}/{second_transition} and {first_direction}/{second_direction}: {str(e)}")

print(f"Processing complete! Videos saved in {output_base_folder}")
```

#### 结合 image to svg:
#### https://huggingface.co/spaces/svjack/image-to-vector
#### 使用svg 逐步生成视频（要求必须整装）
#### pip install opencv-python "httpx[socks]" CairoSVG
```python
import pandas as pd
from io import BytesIO
from PIL import Image
import cairosvg
import os
import cv2
import numpy as np

def clean_svg(svg_string):
    """Optional function to clean SVG if needed"""
    # Add your SVG cleaning logic here if needed
    return svg_string

def rasterize_svg(svg_string, resolution=1024, dpi=128, scale=2):
    """Convert SVG string to PNG image"""
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=resolution,
            output_height=resolution,
            dpi=dpi,
            scale=scale)
        svg_raster = Image.open(BytesIO(svg_raster_bytes))
    except:
        try:
            svg = clean_svg(svg_string)
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=resolution,
                output_height=resolution,
                dpi=dpi,
                scale=scale)
            svg_raster = Image.open(BytesIO(svg_raster_bytes))
        except:
            svg_raster = Image.new('RGB', (resolution, resolution), color='white')
    return svg_raster

def process_svg_to_video(input_svg_path, output_video_path, video_duration_seconds=10, resolution=224, chunk_size=100):
    """Process SVG file and create a video with specified duration"""
    # Read SVG file
    df = pd.read_table(input_svg_path, header=None)
    df_head = df.head(3)
    df_tail = df.tail(1)
    df_middle = df.iloc[3:-1, :]

    # Calculate number of chunks
    total_rows = len(df_middle)
    num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    # Create a temporary directory for images
    temp_dir = "temp_video_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Process each chunk and save as image
    frame_files = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Get current chunk
        current_chunk = df_middle.iloc[0:end_idx, :]

        # Combine with head and tail
        combined_df = pd.concat([df_head, current_chunk, df_tail], axis=0)
        svg_content = "\n".join(combined_df[0].values.tolist())

        # Convert to image and save
        img = rasterize_svg(svg_content, resolution=resolution)
        img_filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
        img.save(img_filename)
        frame_files.append(img_filename)

    # Create video from frames
    create_video_from_frames(frame_files, output_video_path, video_duration_seconds, resolution)

    # Clean up temporary files
    for file in frame_files:
        os.remove(file)
    os.rmdir(temp_dir)

    print(f"Video saved to {output_video_path}")

def create_video_from_frames(frame_files, output_path, duration_seconds, resolution):
    """Create video from sequence of frames with specified duration"""
    # Calculate frame rate based on desired duration
    num_frames = len(frame_files)
    fps = num_frames / duration_seconds

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
    video = cv2.VideoWriter(output_path, fourcc, fps, (resolution, resolution))

    # Read each frame and write to video
    for frame_file in frame_files:
        # Read image with PIL and convert to OpenCV format
        pil_img = Image.open(frame_file)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        video.write(cv_img)

    # Add last frame to fill remaining time if needed
    if num_frames > 0:
        remaining_frames = int(fps * duration_seconds) - num_frames
        for _ in range(remaining_frames):
            video.write(cv_img)

    video.release()

# Example usage
input_svg = "svg_output.svg"  # Your input SVG file
output_video = "output_video.mp4"  # Output video file
video_duration = 10  # Desired video duration in seconds

process_svg_to_video(input_svg, output_video, video_duration_seconds=video_duration, resolution = 1024)

```

#### 通过 图片 生成 对应 生成过程 的 gradio 应用 
#### https://huggingface.co/spaces/svjack/image-to-vector-video
