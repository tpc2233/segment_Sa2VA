#mod for cu128 tpc2233
import gradio as gr
import sys

from projects.sa2va.gradio.app_utils import\
    process_markdown, show_mask_pred, description, preprocess_video,\
    show_mask_pred_video, image2video_and_save

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig, GenerationMixin)
import argparse
import os

import cv2 

# ADD THIS HELPER FUNCTION
def load_video_frames(video_path):
    """Loads a video from a file path into a list of RGB numpy frames."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads frames in BGR format, convert to RGB for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args(args):
    parser = argparse.ArgumentParser(description="Sa2VA Demo")
    parser.add_argument('hf_path', help='Sa2VA hf path.')
    return parser.parse_args(args)

def inference(image, video, follow_up, input_str):
    # (The initial part of your function stays the same)
    input_image = image
    if image is not None and (video is not None and os.path.exists(video)):
        return image, video, "Error: Please only input a image or a video !!!"
    if image is None and (video is None or not os.path.exists(video)) and not follow_up:
        return image, video, "Error: Please input a image or a video !!!"

    if not follow_up:
        print('Log: History responses have been removed!')
        global_infos.n_turn = 0
        global_infos.inputs = ''
        text = input_str
        global_infos.image_for_show = image
        global_infos.image = image
        global_infos.video = video
        global_infos.input_type = "image" if image is not None else "video"
    else:
        past_history = global_infos.inputs
        text = f"{past_history}\n\nHuman: {input_str}"
        image = global_infos.image
        video = global_infos.video

    input_type = global_infos.input_type
    
    if "<image>" not in text:
        text = "<image>\n" + text
    
    if input_type == "video":
        video_processed = preprocess_video(video, text)
    else:
        video_processed = None

    if input_type == "image":
        input_dict = {'image': image, 'text': text, 'mask_prompts': None, 'tokenizer': tokenizer}
    else:
        input_dict = {'video': video_processed, 'text': text, 'mask_prompts': None, 'tokenizer': tokenizer}

    return_dict = sa2va_model.predict_forward(**input_dict)
    
    predict = return_dict['prediction'].strip()
    new_history_turn = f"\n\nHuman: {input_str}\n\nAssistant: {predict}"
    global_infos.inputs += new_history_turn


    # Initialize variables with default values before the if/else blocks.
    # This guarantees they always exist.
    image_mask_show = global_infos.image_for_show
    video_mask_show = global_infos.video # This will be either the path or None
    selected_colors = []


    if 'prediction_masks' in return_dict and return_dict['prediction_masks']:
        if input_type == "image":
            # Overwrite the defaults if we have masks for an image
            image_mask_show, selected_colors = show_mask_pred(global_infos.image_for_show, return_dict['prediction_masks'])
        else: # input_type == "video"
            # Overwrite the defaults if we have masks for a video
            image_mask_show = None # For video output, the image pane should be empty
            loaded_frames = load_video_frames(video)
            processed_frames, selected_colors = show_mask_pred_video(loaded_frames, return_dict['prediction_masks'])
            video_mask_show = image2video_and_save(processed_frames, save_path="./ret_video.mp4")
    # If the condition is false, the default values from the initialization step will be used.
    # The old 'else' block is no longer needed because the defaults handle that case.

    global_infos.n_turn += 1
    
    processed_predict = process_markdown(predict, selected_colors)
    
    return image_mask_show, video_mask_show, processed_predict

def init_models(args):
    model_path = args.hf_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        #force_download=True,    
        trust_remote_code=True,
    ).eval().cuda()


    
    # 1. Get a reference to the language model component (InternLM2ForCausalLM)
    language_model = model.language_model
    
    # 2. Get the actual class object for InternLM2ForCausalLM
    LM_Class = type(language_model)
    
    # 3. Check if GenerationMixin is already a base class (it shouldn't be, based on the error)
    if GenerationMixin not in LM_Class.__bases__:
        # 4. Dynamically create a new class that correctly inherits from both
        # This is the safest way to ensure all methods are available.
        # However, a simpler assignment is often sufficient:
        
        # Safest injection: set the base classes directly
        LM_Class.__bases__ += (GenerationMixin,)
        
        # Since this model is loaded with trust_remote_code=True, it often has its own implementation 
        # of key methods (like forward, __init__). We only need the mixin utilities.
        

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    return model, tokenizer



class global_infos:
    inputs = ''
    n_turn = 0
    image_width = 0
    image_height = 0

    image_for_show = None
    image = None
    video = None

    input_type = "image" # "image" or "video"

if __name__ == "__main__":
    # get parse args and set models
    args = parse_args(sys.argv[1:])

    sa2va_model, tokenizer = \
        init_models(args)

    demo = gr.Interface(
        inference,
        inputs=[
            gr.Image(type="pil", label="Upload Image", height=360),
            gr.Video(sources=["upload", "webcam"], label="Upload mp4 video", height=360),
            gr.Checkbox(label="Follow up Question"),
            gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),],
        outputs=[
            gr.Image(type="pil", label="Output Image"),
            gr.Video(label="Output Video", show_download_button=True, format='mp4'),
            gr.Markdown()],
        theme=gr.themes.Soft(), allow_flagging="auto", description=description,
        title='Sa2VA'
    )

    demo.queue()
    demo.launch(share=True)
