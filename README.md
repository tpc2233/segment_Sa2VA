# Segment Sa2VA: Images and Videos



## Model Zoo

| Model Name |                             Base MLLM                             |                                 Language Part                                 |                       HF Link                        |
|:----------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------:|:----------------------------------------------------:|
|  Sa2VA-4B  | [InternVL2.5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) |    [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-4B) |
|  Sa2VA-26B | [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B) |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-26B) |


## Installation


### Dependencies for Nvidia Blackwell cards
Tested on CUDA 12.8 | Blackwell GPU
```bash
git clone https://github.com/bytedance/segment_Sa2VA.git
cd segment_Sa2VA
conda create -n segment_Sa2VA python=3.12.9 -y
conda activate segment_Sa2VA
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 
pip install -r requirements_blackwell.txt
pip install flash-attn --no-build-isolation
```

## 🤗 Models Download

Automatic download models by running codes for
Low VRAM GPU
```shell
python models_download.py 4B
```
Mid VRAM GPU
```shell
python models_download.py 8B
```
High VRAM GPU
```shell
python models_download.py 26B
```



## 🤗 Quick Start

Using `gradio`. You can try it to build a local chat interface quickly.
Low VRAM GPU
```shell
PYTHONPATH=. python projects/sa2va/gradio/app2.py models_downloads/Sa2VA-4B
```
High VRAM GPU
```shell
PYTHONPATH=. python projects/sa2va/gradio/app3.py models_downloads/Sa2VA-26B
```



**Single-GPU Evaluation Example:**
```bash
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model Sa2VA-1B --verbose
```

**Multi-GPU Evaluation Example:**
```bash
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN SEEDBench_IMG MMStar AI2D_TEST MMMU_DEV_VAL ScienceQA_TEST --model Sa2VA-4B Sa2VA-8B --verbose
```
</details>





