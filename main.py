import argparse
import os
import torch
import logging
from models.sana import get_sana
from models.hidream import get_hidream
from prompt.loader import read_prompt_csv
from prompt.generate import generate_image
from utils.logger import setup_logger
from utils.misc import get_device, get_dtype
from utils.grid import create_grid_image
from PIL import Image

CATEGORY_LIST = [
    "Colors",
    "Conflicting",
    "Counting",
    "DALL-E",
    "Descriptions",
    "Gary Marcus et al.",
    "Misspellings",
    "Positional",
    "Rare Words",
    "Reddit",
    "Text",
]

AVAILABLE_MODELS = {
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers": {
        "type": "sana",
        "num_inference_steps": 2,
    },
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers": {
        "type": "sana",
        "num_inference_steps": 2,
    },
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers": {
        "type": "sana",
        "num_inference_steps": 20,
        "guidance_scale": 4.5,
    },
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers": {
        "type": "sana",
        "num_inference_steps": 20,
        "guidance_scale": 4.5,
    },
    # https://github.com/HiDream-ai/HiDream-I1/blob/main/inference.py
    "HiDream-ai/HiDream-I1-Fast": {
        "type": "hidream",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
    },
    "HiDream-ai/HiDream-I1-Dev": {
        "type": "hidream",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
    },
    "HiDream-ai/HiDream-I1-Full": {
        "type": "hidream",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
    },
}

parser = argparse.ArgumentParser(description="Prompt Loader")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)
# prompt related
parser.add_argument(
    "--prompt", type=str, default="DrawBench.csv", help="Path to the prompt CSV file"
)
parser.add_argument(
    "--category",
    type=str,
    nargs="+",
    default=["Colors"],
    help="Categories to load prompts from (multiple categories can be specified)",
)
# 새로운 플래그 추가: 모든 카테고리 사용
parser.add_argument(
    "--all-categories",
    action="store_true",
    help="Process all available categories in the prompt file",
)
parser.add_argument(
    "--num", type=int, default=1, help="Number of prompts to load from each category"
)
# model related
parser.add_argument(
    "--repo-id",
    type=str,
    default="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    help="Model repository ID",
)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["float16", "bfloat16", "float32"],
    help="Data type for the model",
)
# output related
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs",
    help="Directory to save generated images",
)
# grid related
parser.add_argument(
    "--no_grid",
    action="store_true",
    help="Skip creating grid images (grids are created by default)",
)
parser.add_argument(
    "--grid_rows",
    type=int,
    default=None,
    help="Number of rows in the grid (if not specified, square grid will be created)",
)
parser.add_argument(
    "--title_size",
    type=int,
    default=100,
    help="Height of the title area in pixels",
)
parser.add_argument(
    "--prompt_size",
    type=int,
    default=80,
    help="Height of the prompt area in pixels",
)
parser.add_argument(
    "--title_font_size",
    type=int,
    default=36,
    help="Font size for the category title",
)
parser.add_argument(
    "--prompt_font_size",
    type=int,
    default=18,
    help="Font size for the prompts",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)

if __name__ == "__main__":
    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    logger.info("Loading prompt file: %s", args.prompt)
    prompt_dict = read_prompt_csv(args.prompt)

    device = get_device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = get_dtype(args.dtype)

    # 모든 카테고리 선택 플래그가 활성화되었으면 prompt_dict의 모든 키를 사용
    if args.all_categories:
        categories_to_process = list(prompt_dict.keys())
        logger.info(
            "Processing all available categories: %s", ", ".join(categories_to_process)
        )
    else:
        categories_to_process = args.category

    # Get prompts from requested categories
    selected_prompts = {}
    for category in categories_to_process:
        if category in prompt_dict:
            # Get only 'num' prompts (or fewer if category has less)
            selected_prompts[category] = prompt_dict[category][
                : min(args.num, len(prompt_dict[category]))
            ]
            if args.verbose:
                logger.info(
                    "Loaded %d prompts from category '%s'",
                    len(selected_prompts[category]),
                    category,
                )
        else:
            logger.warning("Category '%s' not found in prompt file", category)

    # Log selected prompts
    if args.verbose:
        for category, prompts in selected_prompts.items():
            logger.info("Category: %s", category)
            for i, prompt in enumerate(prompts, 1):
                logger.info(" %d. %s", i, prompt)

    # 모델 선택 및 이미지 생성 부분 개선 (중첩 if문 제거)
    if args.repo_id not in AVAILABLE_MODELS:
        logger.error("Repository ID not in supported list: %s", args.repo_id)
        exit(1)

    # 모델 구성 가져오기
    configs = AVAILABLE_MODELS[args.repo_id].copy()
    model_type = configs.pop("type")  # 타입 추출 및 제거

    if model_type == "sana":
        # SANA 모델 로드
        pipeline = get_sana(
            repo_id=args.repo_id,
            device=device,
            dtype=dtype,
        )

    elif model_type == "hidream":
        # HiDream 모델 로드
        pipeline = get_hidream(
            repo_id=args.repo_id,
            device=device,
            dtype=dtype,
            shift=configs.pop("shift", 3.0),
        )

    # 모델 이름 추출
    model_name = os.path.basename(args.repo_id)

    # 기본 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 그리드 생성용 이미지 저장
    category_images = {}
    category_prompts = {}

    # 각 카테고리별 이미지 생성
    for category, prompts in selected_prompts.items():
        # 디렉토리 구조 생성: {model_name}/{category}/
        model_dir = os.path.join(args.output_dir, model_name)
        category_dir = os.path.join(model_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        logger.info("Starting image generation for category '%s'", category)

        # 그리드 생성용 리스트 초기화
        if not args.no_grid:
            category_images[category] = []
            category_prompts[category] = []

        for i, prompt in enumerate(prompts):
            logger.info("Generating image with prompt '%s'...", prompt)

            try:
                # 이미지 생성
                image = generate_image(pipeline, prompt, configs, generator)

                # 프롬프트를 파일명으로 사용하여 이미지 저장
                # 프롬프트를 파일명으로 사용하여 이미지 저장 (최대 24글자로 제한)
                cleaned_prompt = (
                    prompt.replace(" ", "_")
                    .replace(".", "")
                    .replace(",", "")
                    .replace("!", "")
                    .replace("?", "")
                )
                if len(cleaned_prompt) > 24:
                    cleaned_prompt = cleaned_prompt[:24]
                image_filename = f"{cleaned_prompt}_seed{args.seed}.png"
                image_path = os.path.join(category_dir, image_filename)
                image.save(image_path)

                # 그리드 생성용으로 저장
                if not args.no_grid:
                    category_images[category].append(image)
                    category_prompts[category].append(prompt)

                logger.info("Image saved: %s", image_path)
            except Exception as e:
                logger.error("Error generating image: %s", str(e))

        logger.info("Completed image generation for category '%s'", category)

        # 그리드 이미지 생성 (--no_grid 옵션이 지정되지 않은 경우)
        if (
            not args.no_grid
            and category in category_images
            and category_images[category]
        ):
            logger.info("Creating grid image for category '%s'", category)
            grid_filename = f"{category}_grid.png"
            grid_path = os.path.join(model_dir, grid_filename)

            try:
                create_grid_image(
                    images=category_images[category],
                    prompts=category_prompts[category],
                    category=category,
                    rows=args.grid_rows,
                    output_path=grid_path,
                    title_size=args.title_size,
                    prompt_size=args.prompt_size,
                    title_font_size=args.title_font_size,
                    prompt_font_size=args.prompt_font_size,
                )
                logger.info("Grid image saved: %s", grid_path)
            except Exception as e:
                logger.error("Error creating grid image: %s", str(e))

    # 모든 카테고리 처리 완료

    logger.info("All processing completed")
