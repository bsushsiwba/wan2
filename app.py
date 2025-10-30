import torch
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

import aoti


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 80

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

aoti.aoti_blocks_load(pipe.transformer, "zerogpu-aoti/Wan2", variant="fp8da")
aoti.aoti_blocks_load(pipe.transformer_2, "zerogpu-aoti/Wan2", variant="fp8da")


default_prompt_i2v = (
    "이 이미지에 생동감을 부여하고, 영화 같은 움직임과 부드러운 애니메이션을 적용"
)
default_negative_prompt = "색조 선명, 과다 노출, 정적, 세부 흐림, 자막, 스타일, 작품, 그림, 화면, 정지, 회색조, 최악 품질, 저품질, JPEG 압축, 추함, 불완전, 추가 손가락, 잘못 그려진 손, 잘못 그려진 얼굴, 기형, 변형, 형태 불량 사지, 손가락 융합, 정지 화면, 지저분한 배경, 세 개의 다리, 배경 사람 많음, 뒤로 걷기"


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
        raise gr.Error("이미지를 업로드해주세요.")

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


# 세련된 한글 UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 WAN 기반 초고속 이미지 to 비디오 무료 생성 오픈소스")
    gr.Markdown("** WAN 2.2 14B + FAST + 한글화 + 튜닝 ** - 4~8단계로 빠른 영상 생성")
    gr.Markdown("** 트래픽 제한시 다음 4개의 미러링 서버들을 이용하여 분산 사용 권고")

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
            input_image_component = gr.Image(type="pil", label="입력 이미지")
            prompt_input = gr.Textbox(
                label="프롬프트", value=default_prompt_i2v, lines=2
            )
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION,
                maximum=MAX_DURATION,
                step=0.1,
                value=3.5,
                label="영상 길이 (초)",
            )

            with gr.Accordion("고급 설정", open=False):
                negative_prompt_input = gr.Textbox(
                    label="네거티브 프롬프트", value=default_negative_prompt, lines=2
                )
                steps_slider = gr.Slider(
                    minimum=1, maximum=30, step=1, value=6, label="생성 단계"
                )
                guidance_scale_input = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.5,
                    value=1,
                    label="가이던스 스케일 1",
                )
                guidance_scale_2_input = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.5,
                    value=1,
                    label="가이던스 스케일 2",
                )
                seed_input = gr.Slider(
                    label="시드", minimum=0, maximum=MAX_SEED, step=1, value=42
                )
                randomize_seed_checkbox = gr.Checkbox(
                    label="랜덤 시드 사용", value=True
                )

            generate_button = gr.Button("🎥 영상 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(
                label="생성된 영상", autoplay=True, interactive=False
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
                "POV 셀카 영상, 선글라스 낀 흰 고양이가 서핑보드에 서서 편안한 미소. 배경에 열대 해변(맑은 물, 녹색 언덕, 구름 낀 푸른 하늘). 서핑보드가 기울어지고 고양이가 바다로 떨어지며 카메라가 거품과 햇빛과 함께 물속으로 빠짐. 잠깐 물속에서 고양이 얼굴 보이다가 다시 수면 위로 올라와 셀카 촬영 계속, 즐거운 여름 휴가 분위기.",
                4,
            ],
            [
                "wan22_input_2.jpg",
                "세련된 달 탐사 차량이 왼쪽에서 오른쪽으로 미끄러지듯 이동하며 달 먼지를 일으킴. 흰 우주복을 입은 우주인들이 달 특유의 뛰는 동작으로 탑승. 먼 배경에서 VTOL 비행체가 수직으로 하강하여 표면에 조용히 착륙. 장면 전체에 걸쳐 초현실적인 오로라가 별이 가득한 하늘을 가로지르며 춤추고, 녹색, 파란색, 보라색 빛의 커튼이 달 풍경을 신비롭고 마법 같은 빛으로 감쌈.",
                4,
            ],
            [
                "kill_bill.jpeg",
                "우마 서먼의 캐릭터 베아트릭스 키도가 영화 같은 조명 속에서 날카로운 카타나 검을 안정적으로 들고 있음. 갑자기 광택 나는 강철이 부드러워지고 왜곡되기 시작하며 가열된 금속처럼 구조적 완전성을 잃기 시작. 검날의 완벽한 끝이 천천히 휘어지고 늘어지며, 녹은 강철이 은빛 물줄기로 아래로 흘러내림. 변형은 처음에는 미묘하게 시작되다가 금속이 점점 더 유동적이 되면서 가속화. 카메라는 그녀의 얼굴을 고정하고 날카로운 눈빛이 점차 좁아지는데, 치명적인 집중이 아니라 무기가 눈앞에서 녹는 것을 보며 혼란과 경악. 호흡이 약간 빨라지며 이 불가능한 변형을 목격. 녹는 현상이 강화되고 카타나의 완벽한 형태가 점점 추상적이 되며 손에서 수은처럼 떨어짐. 녹은 방울이 부드러운 금속 충격음과 함께 바닥에 떨어짐. 표정이 차분한 준비에서 당혹감과 우려로 바뀌며 전설적인 복수의 도구가 손에서 문자 그대로 액화되어 무방비 상태가 됨.",
                6,
            ],
        ],
        inputs=[input_image_component, prompt_input, steps_slider],
        outputs=[video_output, seed_input],
        fn=generate_video,
        cache_examples="lazy",
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
