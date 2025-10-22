# Segment Sa2VA: Images and Videos



## Model Zoo

| Model Name |                             Base MLLM                             |                                 Language Part                                 |                       HF Link                        |
|:----------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------:|:----------------------------------------------------:|
|  Sa2VA-4B  | [InternVL2.5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) |    [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-4B) |
|  Sa2VA-26B | [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B) |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-26B) |


## Installation


### Dependencies for CUDA 12.8  
Tested on Nvidia Blackwell cards
```bash
git clone https://github.com/bytedance/segment_Sa2VA.git
cd segment_Sa2VA
conda create -n segment_Sa2VA python=3.12.9 -y
conda activate segment_Sa2VA
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 
pip install -r requirements_blackwell.txt
pip install flash-attn --no-build-isolation
```

## Models Download

Automatic download models by running codes for

Low VRAM GPU 12+
```shell
python models_download.py 4B
```
Mid VRAM GPU 24+
```shell
python models_download.py 8B
```
High VRAM GPU 80+
```shell
python models_download.py 26B
```



## 🤗 Quick Start

Using `gradio`. You can try it to build a local chat interface quickly.

Low VRAM GPU
```shell
PYTHONPATH=. python projects/sa2va/gradio/app3.py models_downloads/Sa2VA-4B
```
High VRAM GPU
```shell
PYTHONPATH=. python projects/sa2va/gradio/app3.py models_downloads/Sa2VA-26B
```


### Examples


Prompt: 
Can you please segment cheetah in the given image
![cheetah-618a266c098a05](https://github.com/user-attachments/assets/c6969ee7-5ab2-4c53-a77b-3f746f6b365f)

![cheetah](https://github.com/user-attachments/assets/80fc991f-0144-4c39-ba9b-28a0b9b092ea)

Prompt: 
Can you please segment pink toy in the given image

![pink-6a4307bace4d15](https://github.com/user-attachments/assets/b6329b68-d9af-42e5-aab6-0aed97c4a51b)

Prompt: 
Can you please segment purple toy in the given image
![purple-64ad180f22b3c8](https://github.com/user-attachments/assets/20dcf358-c009-4ae1-af77-ba1690af69bb)

Prompt: 
Can you please segment main person in the given image
![man-6908e430400799](https://github.com/user-attachments/assets/38d4dca3-3961-4c17-ab74-db391db4df54)




