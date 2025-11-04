import torch

torch.backends.cuda.preferred_linalg_library("magma")

from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import tempfile
import numpy as np
from PIL import Image
import random
import gc
import os

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 1024
MIN_DIM = 768
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 240
MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)


# -----------------------------
# Load Model and Adapters
# -----------------------------
print("Loading model...")
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
    weight_name="Lightx2v/lightx2v_I2V_14B_720p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v",
)
kwargs_lora = {}
kwargs_lora["load_into_transformer_2"] = True
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_720p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2",
    **kwargs_lora
)
pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3.0, components=["transformer"])
pipe.fuse_lora(
    adapter_names=["lightx2v_2"], lora_scale=1.0, components=["transformer_2"]
)
pipe.unload_lora_weights()

# Quantize models
print("Quantizing models...")
quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())


default_prompt_i2v = (
    "이 이미지에 생동감을 부여하고, 영화 같은 움직임과 부드러운 애니메이션을 적용"
)


# -----------------------------
# Helper Functions
# -----------------------------
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


def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt="",
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
):

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    print(f"Generating {num_frames} frames with seed={current_seed}...")
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

    output_path = "output.mp4"
    export_to_video(output_frames_list, output_path, fps=FIXED_FPS)
    print(f"✅ Video saved as {output_path}")
    return output_path


# -----------------------------
# Main
# -----------------------------
def main():
    sample_image_path = "sample.jpg"
    if not os.path.exists(sample_image_path):
        raise FileNotFoundError(f"Sample image not found at {sample_image_path}")

    input_image = Image.open(sample_image_path).convert("RGB")
    prompt = "cat jumping and playing"
    negative_prompt = "색조 선명, 과다 노출, 정적, 세부 흐림, 자막, 스타일, 작품, 그림, 화면, 정지, 회색조, 최악 품질, 저품질, JPEG 압축, 추함, 불완전, 추가 손가락, 잘못 그려진 손, 잘못 그려진 얼굴, 기형, 변형, 형태 불량 사지, 손가락 융합, 정지 화면, 지저분한 배경, 세 개의 다리, 배경 사람 많음, 뒤로 걷기"

    generate_video(
        input_image=input_image,
        prompt=prompt,
        steps=6,
        negative_prompt=negative_prompt,
        duration_seconds=4.0,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=42,
        randomize_seed=False,
    )


if __name__ == "__main__":
    main()
