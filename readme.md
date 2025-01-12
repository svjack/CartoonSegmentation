# CartoonSegmentation

Implementations of the paper _Instance-guided Cartoon Editing with a Large-scale Dataset_, including an instance segmentation for cartoon/anime characters and some visual techniques built around it.


[![arXiv](https://img.shields.io/badge/arXiv-2312.01943-<COLOR>)](http://arxiv.org/abs/2312.01943)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CartoonSegmentation/CartoonSegmentation/blob/main/run_in_colab.ipynb)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://cartoonsegmentation.github.io/)

<p float="center">
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser00.jpg" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser01.jpg" width="24%" />
  <img src="https://github.com/CartoonSegmentation/CartoonSegmentation/assets/51270320/10301ee4-09c1-45a9-8672-7e0a3cbd1c20" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser03.jpg" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser10.jpg" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser11.jpg" width="24%" />
  <img src="https://github.com/CartoonSegmentation/CartoonSegmentation/assets/51270320/602f8e5b-bec2-4f07-af50-b72d6411da70" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser13.jpg" width="24%" />
</p>



## Preperation

### Install Dependencies

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

git clone https://github.com/svjack/CartoonSegmentation && cd CartoonSegmentation
#conda env create -f conda_env.yaml
conda create --name animeins python=3.10
conda activate animeins
pip install ipykernel
python -m ipykernel install --user --name animeins --display-name "animeins"
pip install -r requirements.txt

pip install torch==2.1.1 torchvision
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmdet
pip install "numpy<2.0.0"
pip install moviepy==1.0.3
pip install "httpx[socks]"
```

```bash
pip install cupy-cuda12x
```

```bash
mkdir -p data/libs

sudo apt install build-essential libopencv-dev -y
git clone https://github.com/AnimeIns/PyPatchMatch && cd PyPatchMatch

mkdir release && cd release
sudo apt install cmake
cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ../..
mv PyPatchMatch/release/libpatchmatch_inpaint.so ./data/libs
rm -rf PyPatchMatch

```

```bash
huggingface-cli lfs-enable-largefiles .
mkdir models
git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentatio
```



- Run 3d Kenburns Demo
```bash
python run_kenburns.py --cfg configs/3dkenburns.yaml --input-img examples/kenburns_lion.png
```

- Run 3d Kenburns On Genshin Impact Image Demo
```bash
python run_kenburns_batch.py --cfg configs/3dkenburns.yaml --input-img Genshin_Impact_Images --save_dir Genshin_Impact_Images_3dkenburns
```


https://github.com/user-attachments/assets/8b511789-81cb-49b2-a0ad-62f1ffb35a3b




https://github.com/user-attachments/assets/82bed331-bbd9-4513-a89a-50aa4ab9b645




https://github.com/user-attachments/assets/c9b7c663-406d-4cd5-9531-44849f668da6



https://github.com/user-attachments/assets/17d746b4-541f-46d8-9e44-6880e24b1d4b





https://github.com/user-attachments/assets/02b9c9b8-391d-40c3-863c-5b475716f0bd

```bash
python run_kenburns_batch.py --cfg configs/3dkenburns_no_depth_field.yaml --input-img Genshin_Impact_Images --save_dir Genshin_Impact_Images_3dkenburns_no_depth_field
```

https://github.com/user-attachments/assets/34775b11-f42b-4b1d-8e98-14dfc23f6364




https://github.com/user-attachments/assets/e2d8b068-9434-40ca-855b-18f3523dd25d




https://github.com/user-attachments/assets/cf92bfcf-4dc7-42b1-a20a-223f5e140b76



### After above install of gpu version
- Use Segmentation Demo on gpu
```bash
git clone https://huggingface.co/spaces/svjack/AnimeIns_CPU && cd AnimeIns_CPU
python app.py
```

- Use script to run Segmentation Demo
```bash
python seg_script.py Genshin_Impact_Images Genshin_Impact_Images_Seg

cd ..
featurize dataset download bad53b75-692d-422a-8262-f3f07e3aab81
featurize dataset download b0190045-da94-40ac-b267-2e2efd68b2cb
unzip 原神单人图片2.zip
unzip 原神单人图片1.zip
cd  AnimeIns_CPU

python seg_script.py ../single_output_images_v2 Genshin_Impact_Images_Seg_v2
python seg_script.py ../single_output_images Genshin_Impact_Images_Seg_v1
```

### Download models

```bash
huggingface-cli lfs-enable-largefiles .
mkdir models
git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentation
```

# Genshin Impact 3d Kenburns Manga Demo use CartoonSegmentation

# Genshin Impact Manga Processing Workflow

## 1. Install Dependencies

First, install the required Python libraries:

```bash
!pip install datasets
```

## 2. Load the Dataset

Load the Genshin Impact Manga dataset:

```python
from datasets import load_dataset

ds = load_dataset("svjack/Genshin-Impact-Manga")["train"]
```

## 3. Save Manga Images

Save the images from the dataset to a local directory `manga_save`:

```python
from tqdm import tqdm
import os

os.makedirs("manga_save", exist_ok=True)
l = len(ds)

for i in tqdm(range(l)):
    d = ds[i]
    sTitle = d["sTitle"]
    name = d["name"]
    img = d["image"]
    save_path = os.path.join("manga_save", sTitle.replace(" ", "_") + "_" + name).replace(".jpg", ".png")
    img.resize((800, 1131)).save(save_path)
```

## 4. Analyze Image Sizes

Analyze the sizes of the saved images:

```python
import pathlib
import pandas as pd
from PIL import Image

df = pd.DataFrame(
    pd.Series(pathlib.Path("manga_save").rglob("*.png")).map(
        lambda x: (x, Image.open(x).size)
    ).values.tolist()
)
df.columns = ["path", "im_size"]
df["im_size"].value_counts()
```

## 5. Upscale Images

Use [APISR](https://github.com/svjack/APISR) to upscale the images:

```bash
rm -rf ../CartoonSegmentation/manga_save/.ipynb_checkpoints
python test_code/inference.py --input_dir ../CartoonSegmentation/manga_save  --weight_path pretrained/4x_APISR_GRL_GAN_generator.pth  --store_dir manga_save_4x
```

## 6. Add Transparent Borders

Add transparent borders to the upscaled images:

```python
from PIL import Image
import os
from tqdm import tqdm

def add_inner_transparent_border(input_path, output_path, border_width, output_width, output_height):
    for filename in tqdm(os.listdir(input_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_path, filename)
            img = Image.open(img_path).convert("RGBA")

            original_width, original_height = img.size
            new_width = original_width - 2 * border_width
            new_height = original_height - 2 * border_width

            if new_width <= 0 or new_height <= 0:
                raise ValueError("Border width is too large to shrink the image.")

            inner_img = img.resize((new_width, new_height))
            new_img = Image.new("RGBA", (original_width, original_height), (0, 0, 0, 0))

            paste_x = (original_width - new_width) // 2
            paste_y = (original_height - new_height) // 2

            new_img.paste(inner_img, (paste_x, paste_y), inner_img)
            final_img = new_img.resize((output_width, output_height))

            output_img_path = os.path.join(output_path, filename)
            os.makedirs(output_path, exist_ok=True)
            final_img.save(output_img_path)

input_path = "manga_save_4x"
output_path = "manga_save_4x_pad_512"

border_width = 512  # Set border width
output_width = 2816  # Set output image width
output_height = 4096  # Set output image height

add_inner_transparent_border(input_path, output_path, border_width, output_width, output_height)
```

## 7. Add 3d Kenburns Effects

Use the `run_kenburns_batch.py` script to add 3d Kenburns effects:

```bash
python run_kenburns_batch.py --cfg configs/3dkenburns_no_depth_field_max_reso.yaml --input-img manga_save_4x_pad_512 --save_dir manga_save_4x_3dkenburns_no_depth_field_512
```

## 8. Reorganize Resulting Videos

Reorganize the generated videos by chapter:

```python
import os
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import shutil

ds = load_dataset("svjack/Genshin-Impact-Manga")["train"]
df = ds.to_pandas()

input_path = "manga_save_4x_3dkenburns_no_depth_field_512_mix"
output_path = "reorganized_manga"

os.makedirs(output_path, exist_ok=True)

grouped = df.groupby("sTitle")

for sTitle, group in tqdm(grouped):
    subdir = os.path.join(output_path, sTitle.replace(" ", "_"))
    os.makedirs(subdir, exist_ok=True)

    for idx, row in group.iterrows():
        name = row["name"]
        mp4_filename = f"{sTitle}_{name}_4x.mp4".replace(" ", "_").replace(".jpg", "").replace(".png", "")
        mp4_filepath = os.path.join(input_path, mp4_filename)

        if os.path.exists(mp4_filepath):
            new_filename = name.replace(".jpg", ".mp4").replace(".png", ".mp4")
            new_filepath = os.path.join(subdir, new_filename)
            shutil.copy2(mp4_filepath, new_filepath)
        else:
            print(f"File does not exist: {mp4_filepath}")
```

## 9. Use the Hugging Face Hub Demo Local Directly

Clone and run the demo from Hugging Face Hub:

```bash
git clone https://huggingface.co/spaces/svjack/Genshin-Impact-3d-Kenburns-Manga && cd Genshin-Impact-3d-Kenburns-Manga && pip install -r requirements.txt 
python app.py
```


https://github.com/user-attachments/assets/7570919f-a1b4-41f8-ba0d-1d968223cde5





## Run Segmentation

See `run_segmentation.ipynb``. 

Besides, we have prepared a simple [Huggingface Space](https://huggingface.co/spaces/ljsabc/AnimeIns_CPU) for you to test with the segmentation on the browser. 

![A workable demo](https://animeins.oss-cn-shenzhen.aliyuncs.com/imas.jpg)
*Copyright BANDAI NAMCO Entertainment Inc., We believe this is a fair use for research and educational purpose only.*


## Run 3d Kenburns


https://github.com/dmMaze/CartoonSegmentation/assets/51270320/503c87c3-39d7-40f8-88f9-3ead20e1e5c5



Install cupy following https://docs.cupy.dev/en/stable/install.html  

```bash
pip install cupy-cuda12x
```

Run
``` python
python run_kenburns.py --cfg configs/3dkenburns.yaml --input-img examples/kenburns_lion.png
```
or with the interactive interface:
``` python
python naive_interface.py --cfg configs/3dkenburns.yaml
```
and open http://localhost:8080 in your browser.

Please read configs/3dkenburns.yaml for more advanced settings.  

To use Marigold as depth estimator, run
```
git submodule update --init --recursive
```
and set ```depth_est``` to ```marigold``` in configs/3dkenburns.yaml


### Better Inpainting using Stable-diffusion

To get better inpainting results with Stable-diffusion, you need to install stable-diffusion-webui first, and download the tagger: 
``` bash
git clone https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2 models/wd-v1-4-swinv2-tagger-v2
```

If you're on Windows, download compiled libs from https://github.com/AnimeIns/PyPatchMatch/releases/tag/v1.0 and save them to data/libs, otherwise, you need to compile patchmatch in order to run 3dkenburns or style editing:

#### Compile Patchmatch

``` bash
mkdir -P data/libs
apt install build-essential libopencv-dev -y
git clone https://github.com/AnimeIns/PyPatchMatch && cd PyPatchMatch

mkdir release && cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ../..
mv PyPatchMatch/release/libpatchmatch_inpaint.so ./data/libs
rm -rf PyPatchMatch
```
<i>If you have activated conda and encountered `GLIBCXX_3.4.30' not found or libpatchmatch_inpaint.so: cannot open shared object file: No such file or directory, follow the solution here https://askubuntu.com/a/1445330 </i>

Launch the stable-diffusion-webui with argument `--api` and set the base model to `sd-v1-5-inpainting`, modify `inpaint_type: default` to `inpaint_type: ldm` in configs/3dkenburns.yaml.   

Finally, run 3dkenburns with pre-mentioned commands.


## Run Style Editing
We are using stable-diffusion-webui @ [bef51aed](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/bef51aed032c0aaa5cfd80445bc4cf0d85b408b5) and sd-webui-controlnet @ [aa2aa81](https://github.com/Mikubill/sd-webui-controlnet/commit/aa2aa812e86a1f47ef360572888d66027d640f60).  
It also requires stable-diffusion-webui, patchmatch, and the danbooru tagger, so please follow the `Run 3d Kenburns` and download/install these first.  
Download [sd_xl_base_1.0_0.9vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors), style [lora](https://civitai.com/models/124347/xlmoreart-full-xlreal-enhancer) and [diffusers_xl_canny_mid](https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors) and save them to corresponding directory in stable-diffusion-webui, launch stable-diffusion-webui with argument `--argment` and set `sd_xl_base_1.0_0.9vae` as base model, then run

```
python run_style.py --img_path examples/kenburns_lion.png --cfg configs/3d_pixar.yaml
```
set `onebyone` to False in configs/3d_pixar.yaml to disable instance-aware style editing.


## Run Web UI (Including both _3D Ken Burns_ and _Style Editing_), based on Gradio
All required libraries and configurations have been included, now we just need to execute the Web UI from its Launcher: 

```
python Web_UI/Launcher.py
```
In default configurations, you can find the Web UI here:
- http://localhost:1234 in local
- A random temporary public URL generated by Gradio, such like this: https://1ec9f82dc15633683e.gradio.live
