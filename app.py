import torch

torch.backends.cuda.preferred_linalg_library("magma")

from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 240

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)


pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v",
)
kwargs_lora = {}
kwargs_lora["load_into_transformer_2"] = True
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2",
    **kwargs_lora
)
pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3.0, components=["transformer"])
pipe.fuse_lora(
    adapter_names=["lightx2v_2"], lora_scale=1.0, components=["transformer_2"]
)
pipe.unload_lora_weights()

quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())


default_prompt_i2v = (
    "Bring this image to life with cinematic motion and smooth animation."
)
default_negative_prompt = "Sharp tone, overexposure, static, blurred details, subtitles, style, artwork, painting, screen, still, grayscale, worst quality, low quality, JPEG compression, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, malformed, deformed, misshapen limbs, fused fingers, still frame, messy background, three legs, crowded background, walking backward."


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size

    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height

    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def get_num_frames(duration_seconds: float):
    return 1 + int(
        np.clip(
            int(round(duration_seconds * FIXED_FPS)),
            MIN_FRAMES_MODEL,
            MAX_FRAMES_MODEL,
        )
    )


def get_duration(
    input_image,
    prompt,
    steps,
    negative_prompt,
    duration_seconds,
    guidance_scale,
    guidance_scale_2,
    seed,
    randomize_seed,
    progress,
):
    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 15
    width, height = resize_image(input_image).size
    frames = get_num_frames(duration_seconds)
    factor = frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor**1.5
    return 10 + int(steps) * step_duration


def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    return video_path, current_seed


# Polished English UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 WAN-based Ultra-Fast Image to Video Free Open Source")
    gr.Markdown("** WAN 2.2 14B + FAST + English Localization + Tuning ** - Fast video generation in 4-8 steps")
    gr.Markdown("** If traffic is limited, use the following 4 mirroring servers for distributed usage **")

    gr.HTML(
        """
    <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin: 20px 0;">
        <a href="https://huggingface.co/spaces/Heartsync/wan2_2-I2V-14B-FAST" target="_blank">
            <img src="https://img.shields.io/static/v1?label=WAN%202.2%2014B%20FAST%2B&message=Image%20to%20Video&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=white&style=for-the-badge" alt="badge">
        </a>
        <a href="https://huggingface.co/spaces/ginipick/wan2_2-I2V-14B-FAST" target="_blank">
            <img src="https://img.shields.io/static/v1?label=WAN%202.2%2014B%20FAST%2B&message=Image%20to%20Video&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=white&style=for-the-badge" alt="badge">
        </a>
        <a href="https://huggingface.co/spaces/ginigen/wan2_2-I2V-14B-FAST" target="_blank">
            <img src="https://img.shields.io/static/v1?label=WAN%202.2%2014B%20FAST%2B&message=Image%20to%20Video&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=white&style=for-the-badge" alt="badge">
        </a>
        <a href="https://huggingface.co/spaces/VIDraft/wan2_2-I2V-14B-FAST" target="_blank">
            <img src="https://img.shields.io/static/v1?label=WAN%202.2%2014B%20FAST%2B&message=Image%20to%20Video&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=white&style=for-the-badge" alt="badge">
        </a>
        <a href="https://discord.gg/openfreeai" target="_blank">
            <img src="https://img.shields.io/static/v1?label=Discord&message=Openfree%20AI&color=%230000ff&labelColor=%23800080&logo=discord&logoColor=white&style=for-the-badge" alt="badge"></a>        
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image_component = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(
                label="Prompt", value=default_prompt_i2v, lines=2
            )
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION,
                maximum=MAX_DURATION,
                step=0.1,
                value=3.5,
                label="Video Duration (seconds)",
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt", value=default_negative_prompt, lines=2
                )
                steps_slider = gr.Slider(
                    minimum=1, maximum=30, step=1, value=6, label="Generation Steps"
                )
                guidance_scale_input = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.5,
                    value=1,
                    label="Guidance Scale 1",
                )
                guidance_scale_2_input = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.5,
                    value=1,
                    label="Guidance Scale 2",
                )
                seed_input = gr.Slider(
                    label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42
                )
                randomize_seed_checkbox = gr.Checkbox(
                    label="Use Random Seed", value=True
                )

            generate_button = gr.Button("🎥 Generate Video", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(
                label="Generated Video", autoplay=True, interactive=False
            )

    ui_inputs = [
        input_image_component,
        prompt_input,
        steps_slider,
        negative_prompt_input,
        duration_seconds_input,
        guidance_scale_input,
        guidance_scale_2_input,
        seed_input,
        randomize_seed_checkbox,
    ]
    generate_button.click(
        fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input]
    )

    gr.Examples(
        examples=[
            [
                "wan_i2v_input.JPG",
                "POV selfie video, a white cat with sunglasses standing on a surfboard with a relaxed smile. Tropical beach in the background (clear water, green hills, cloudy blue sky). Surfboard tilting and cat falling into the sea, camera submerging into the water with bubbles and sunlight. Briefly showing the cat's face underwater before resurfacing and continuing the selfie, conveying a joyful summer vacation vibe.",
                4,
            ],
            [
                "wan22_input_2.jpg",
                "A sleek lunar rover gliding from left to right, kicking up lunar dust. Astronauts in white space suits boarding with the Moon's characteristic hopping motion. A VTOL aircraft quietly descending and landing on the surface in the distant background. An ultra-realist
