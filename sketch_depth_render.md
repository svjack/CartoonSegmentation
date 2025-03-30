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
