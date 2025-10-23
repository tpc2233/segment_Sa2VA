#mod for cu128 tpc2233
#added vit and alpha in outs

import gradio as gr
import sys
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationMixin, 
                          VitMatteForImageMatting, VitMatteImageProcessor)
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# PART 1: UTILITIES
# ==============================================================================

from app_utils import (
    process_markdown, show_mask_pred, description, preprocess_video,
    show_mask_pred_video
)

def load_video_frames_and_fps(video_path):
    frames, cap = [], cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise gr.Error(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS); fps = 30 if fps == 0 else fps
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps

def robust_image2video_and_save(frames, save_path, fps):
    if not frames: return None
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame_rgb in frames: out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    out.release()
    return save_path

def create_binary_mask(pred_masks, height, width):
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask_item in pred_masks:
        mask_np = mask_item.cpu().numpy().squeeze().astype(np.uint8) if isinstance(mask_item, torch.Tensor) else mask_item.squeeze().astype(np.uint8)
        if mask_np.shape != (height, width): mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
        combined_mask[mask_np > 0] = 255
    return Image.fromarray(combined_mask)

def create_binary_mask_video(pred_masks, frames, save_path, fps):
    if not frames or not pred_masks: return None
    h, w, _ = frames[0].shape
    mask_frames = []
    for i in range(len(frames)):
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_tensor in pred_masks:
            if i < len(mask_tensor):
                mask_np = mask_tensor[i].cpu().numpy().astype(np.uint8) if isinstance(mask_tensor[i], torch.Tensor) else mask_tensor[i].astype(np.uint8)
                if mask_np.shape != (h, w): mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask[mask_np > 0] = 255
        mask_frames.append(cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB))
    return robust_image2video_and_save(mask_frames, save_path, fps)

def mask_to_trimap(mask, erode_kernel_size, dilate_kernel_size):
    mask_np = np.array(mask.convert("L"))
    erode_k_size, dilate_k_size = max(1, int(erode_kernel_size)//2*2+1), max(1, int(dilate_kernel_size)//2*2+1)
    erode_kernel, dilate_kernel = np.ones((erode_k_size, erode_k_size), np.uint8), np.ones((dilate_k_size, dilate_k_size), np.uint8)
    eroded, dilated = cv2.erode(mask_np, erode_kernel, iterations=1), cv2.dilate(mask_np, dilate_kernel, iterations=1)
    if not np.any(eroded): eroded = mask_np.copy()
    trimap = np.zeros_like(mask_np); trimap[dilated==255]=128; trimap[eroded==255]=255
    return Image.fromarray(trimap)

# ==============================================================================
# PART 2: MODEL INITIALIZATION & GLOBAL STATE
# ==============================================================================
def parse_args(args):
    parser = argparse.ArgumentParser(description="Sa2VA + ViTMatte Demo"); parser.add_argument('hf_path', help='Sa2VA hf path.'); return parser.parse_args(args)

def init_sa2va_model(args):
    model_path = args.hf_path
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
    language_model = model.language_model; LM_Class = type(language_model)
    if GenerationMixin not in LM_Class.__bases__: LM_Class.__bases__ += (GenerationMixin,)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def init_vitmatte_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k").to(device)
    processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
    return model, processor

class global_infos:
    inputs = ''; n_turn = 0; image = None; video = None; input_type = "image"; original_fps = 30

# ==============================================================================
# PART 3: INFERENCE FUNCTIONS
# ==============================================================================
def sa2va_inference(image, video, follow_up, input_str):
    if image is not None and (video is not None and os.path.exists(video)): raise gr.Error("Provide EITHER an image OR a video.")
    if image is None and (video is None or not os.path.exists(video)) and not follow_up: raise gr.Error("Provide an image or a video.")

    if not follow_up:
        global_infos.n_turn, global_infos.inputs = 0, ''; text, global_infos.image, global_infos.video = input_str, image, video
        global_infos.input_type = "image" if image is not None else "video"
    else: text, image, video = f"{global_infos.inputs}\n\nHuman: {input_str}", global_infos.image, global_infos.video

    input_type = global_infos.input_type
    if "<image>" not in text: text = "<image>\n" + text

    if input_type == "video":
        frames, fps = load_video_frames_and_fps(global_infos.video); global_infos.original_fps = fps
        video_processed = preprocess_video(video, text)
        input_dict = {'video': video_processed, 'text': text, 'mask_prompts': None, 'tokenizer': tokenizer}
    else: frames, input_dict = None, {'image': image, 'text': text, 'mask_prompts': None, 'tokenizer': tokenizer}

    return_dict = sa2va_model.predict_forward(**input_dict)
    predict = return_dict['prediction'].strip(); global_infos.inputs += f"\n\nHuman: {input_str}\n\nAssistant: {predict}"

    out_overlay_img, out_overlay_vid, out_mask_img, out_mask_vid = None, None, None, None
    image_for_vitmatte, mask_for_vitmatte, video_path_for_vitmatte, mask_video_path_for_vitmatte = None, None, None, None
    
    if 'prediction_masks' in return_dict and return_dict['prediction_masks']:
        if input_type == "image":
            out_overlay_img, selected_colors = show_mask_pred(global_infos.image, return_dict['prediction_masks'])
            h, w = global_infos.image.height, global_infos.image.width
            out_mask_img = create_binary_mask(return_dict['prediction_masks'], h, w)
            image_for_vitmatte, mask_for_vitmatte = global_infos.image, out_mask_img
        else: # video
            processed_frames, selected_colors = show_mask_pred_video(frames, return_dict['prediction_masks'])
            out_overlay_vid = robust_image2video_and_save(processed_frames, "./ret_video_overlay.mp4", global_infos.original_fps)
            out_mask_vid = create_binary_mask_video(return_dict['prediction_masks'], frames, "./ret_video_mask.mp4", global_infos.original_fps)
            image_for_vitmatte = Image.fromarray(frames[0])
            first_frame_masks = [m[0] for m in return_dict['prediction_masks']]
            mask_for_vitmatte = create_binary_mask(first_frame_masks, image_for_vitmatte.height, image_for_vitmatte.width)
            video_path_for_vitmatte, mask_video_path_for_vitmatte = global_infos.video, out_mask_vid
    else: selected_colors = []; out_overlay_img, out_overlay_vid = global_infos.image, global_infos.video

    processed_predict = process_markdown(predict, selected_colors)
    can_send_to_vitmatte = (image_for_vitmatte is not None or video_path_for_vitmatte is not None)
    return (out_overlay_img, out_overlay_vid, out_mask_img, out_mask_vid, processed_predict, image_for_vitmatte, mask_for_vitmatte, video_path_for_vitmatte, mask_video_path_for_vitmatte, gr.update(interactive=can_send_to_vitmatte))

def update_trimap_preview(mask, erode_size, dilate_size):
    if mask is None: return gr.update(value=None)
    return mask_to_trimap(mask, erode_size, dilate_size)

def send_to_vitmatte(img_state, mask_state, erode_size_in, dilate_size_in):
    if img_state is None or mask_state is None:
        gr.Warning("No image or mask available. Run segmentation first."); return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    initial_trimap = mask_to_trimap(mask_state, erode_size_in, dilate_size_in)
    return (gr.update(value=img_state), gr.update(value=mask_state), gr.update(value=initial_trimap), gr.update(interactive=True), gr.update(selected=1))

def get_refined_alpha_mask(image, mask, erode_size, dilate_size):
    """Core ViT-Matte function that returns only the refined alpha mask."""
    if image is None or mask is None: raise gr.Error("Missing image or mask for get_refined_alpha_mask.")
    
    original_size = image.size # Original (width, height)
    trimap = mask_to_trimap(mask, erode_size, dilate_size)
    inputs = vitmatte_processor(images=image, trimaps=trimap, return_tensors="pt").to(device)
    
    with torch.no_grad(): 
        alphas = vitmatte_model(**inputs).alphas.squeeze().cpu().numpy()
    
    alpha_mask_pil_padded = Image.fromarray((alphas * 255).astype(np.uint8))
    

    original_width, original_height = original_size
    alpha_mask_pil_cropped = alpha_mask_pil_padded.crop((0, 0, original_width, original_height))
    
    return alpha_mask_pil_cropped

def refine_with_vitmatte_wrapper(img_state, mask_state, vid_path_state, mask_vid_path_state, erode_size, dilate_size, progress=gr.Progress(track_tqdm=True)):
    if vid_path_state and mask_vid_path_state:
        progress(0, desc="Loading video frames...")
        original_frames, fps = load_video_frames_and_fps(vid_path_state)
        mask_frames, _ = load_video_frames_and_fps(mask_vid_path_state)
        
        refined_mask_frames_rgb = []
        for orig_frame, mask_frame in tqdm(zip(original_frames, mask_frames), desc="Refining Mask Frame-by-Frame", total=len(original_frames)):
            img_pil = Image.fromarray(orig_frame)
            mask_gray_np = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2GRAY)
            _, binary_mask_np = cv2.threshold(mask_gray_np, 127, 255, cv2.THRESH_BINARY)
            mask_pil = Image.fromarray(binary_mask_np)
            
            alpha_mask_pil = get_refined_alpha_mask(img_pil, mask_pil, erode_size, dilate_size)
            
            alpha_mask_np = np.array(alpha_mask_pil)
            rgb_mask_frame = cv2.cvtColor(alpha_mask_np, cv2.COLOR_GRAY2RGB)
            refined_mask_frames_rgb.append(rgb_mask_frame)
            
        progress(1.0, desc="Stitching frames into refined mask video...")
        output_path = "./ret_video_refined_mask.mp4"
        robust_image2video_and_save(refined_mask_frames_rgb, output_path, fps)
        return None, None, None, output_path

    elif img_state and mask_state:
        progress(0, desc="Starting image matting...")
        alpha_mask = get_refined_alpha_mask(img_state, mask_state, erode_size, dilate_size)
        trimap = mask_to_trimap(mask_state, erode_size, dilate_size)
        
        img_rgba = img_state.convert("RGBA"); img_rgba.putalpha(alpha_mask)
        black_bg = Image.new("RGBA", img_state.size, (0, 0, 0, 255))
        composited_img = Image.alpha_composite(black_bg, img_rgba).convert("RGB")
        
        progress(1.0, desc="Done.")
        return alpha_mask, composited_img, trimap, None
    else:
        raise gr.Error("No valid image or video to process.")

# ==============================================================================
# PART 4: MAIN GRADIO APP LAUNCH
# ==============================================================================
if __name__ == "__main__":
    args = parse_args(sys.argv[1:]); print("Loading Sa2VA model...")
    sa2va_model, tokenizer = init_sa2va_model(args); print("Loading ViT-Matte model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"; vitmatte_model, vitmatte_processor = init_vitmatte_model()
    print("All models loaded.")

    with gr.Blocks(theme=gr.themes.Soft(), title="Sa2VA + ViT-Matte") as demo:
        gr.Markdown(f"<h1 style='text-align: center;'>Segmentation Sa2VA and ViT-Matte</h1>"); gr.Markdown(description)

        shared_image_state, shared_mask_state = gr.State(), gr.State()
        shared_video_path_state, shared_mask_video_path_state = gr.State(), gr.State()

        with gr.Tabs(elem_id="tabs") as tabs_ui:
            with gr.TabItem("Segmentation (Sa2VA)", id=0):
                with gr.Row():
                    with gr.Column():
                        sa2va_img_in, sa2va_vid_in = gr.Image(type="pil", label="Upload Image", height=360), gr.Video(label="Upload MP4 Video", height=360)
                        sa2va_text_in, sa2va_followup_in = gr.Textbox(lines=1, placeholder="e.g., segment the person", label="Text Instruction"), gr.Checkbox(label="Follow-up Question")
                        sa2va_submit_btn = gr.Button("Submit", variant="primary")
                        send_to_vitmatte_btn = gr.Button("Send to ViT-Matte for Refinement", visible=True, interactive=False)
                    with gr.Column():
                        sa2va_md_out = gr.Markdown(label="Response")
                        sa2va_overlay_img_out, sa2va_mask_img_out = gr.Image(type="pil", label="Overlay Output"), gr.Image(type="pil", label="Black and White Mask")
                        sa2va_overlay_vid_out, sa2va_mask_vid_out = gr.Video(label="Overlay Video Output", format='mp4'), gr.Video(label="Black and White Mask Video", format='mp4')
            
            with gr.TabItem("Matting (ViT-Matte)", id=1):
                gr.Markdown("Click 'Send to ViT-Matte' on the first tab. **Note:** Video matting can be very slow!")
                with gr.Row():
                    with gr.Column():
                        vitmatte_img_in = gr.Image(type="pil", label="Source (Image or Video First Frame)", interactive=False)
                        vitmatte_mask_in = gr.Image(type="pil", label="Generated Mask (or Video First Frame)", interactive=False)
                        erode_size_in = gr.Slider(minimum=1, maximum=30, value=10, step=2, label="Erode Kernel Size")
                        dilate_size_in = gr.Slider(minimum=1, maximum=30, value=10, step=2, label="Dilate Kernel Size")
                        vitmatte_submit_btn = gr.Button("Refine with ViT-Matte", variant="primary", interactive=False)
                    with gr.Column():
                        vitmatte_trimap_out = gr.Image(type="pil", label="Generated Trimap (Preview)")
                        vitmatte_alpha_out = gr.Image(type="pil", label="Final Alpha Matte (Image)")
                        vitmatte_fg_out = gr.Image(type="pil", label="Foreground on Black (Image)")
                        vitmatte_video_out = gr.Video(label="Refined Mask Video (Grayscale MP4)", format="mp4")

        sa2va_submit_btn.click(fn=sa2va_inference, inputs=[sa2va_img_in, sa2va_vid_in, sa2va_followup_in, sa2va_text_in], outputs=[sa2va_overlay_img_out, sa2va_overlay_vid_out, sa2va_mask_img_out, sa2va_mask_vid_out, sa2va_md_out, shared_image_state, shared_mask_state, shared_video_path_state, shared_mask_video_path_state, send_to_vitmatte_btn])
        send_to_vitmatte_btn.click(fn=send_to_vitmatte, inputs=[shared_image_state, shared_mask_state, erode_size_in, dilate_size_in], outputs=[vitmatte_img_in, vitmatte_mask_in, vitmatte_trimap_out, vitmatte_submit_btn, tabs_ui])
        vitmatte_submit_btn.click(fn=refine_with_vitmatte_wrapper, inputs=[shared_image_state, shared_mask_state, shared_video_path_state, shared_mask_video_path_state, erode_size_in, dilate_size_in], outputs=[vitmatte_alpha_out, vitmatte_fg_out, vitmatte_trimap_out, vitmatte_video_out])
        erode_size_in.change(fn=update_trimap_preview, inputs=[vitmatte_mask_in, erode_size_in, dilate_size_in], outputs=vitmatte_trimap_out)
        dilate_size_in.change(fn=update_trimap_preview, inputs=[vitmatte_mask_in, erode_size_in, dilate_size_in], outputs=vitmatte_trimap_out)

    demo.queue().launch(share=True)
