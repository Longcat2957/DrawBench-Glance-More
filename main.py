import argparse
import os
import logging
from models.sana import get_sana
from prompt.loader import read_prompt_csv
from prompt.generate import generate_image
from utils.logger import setup_logger
from utils.misc import get_device, get_dtype

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
    }
}

parser = argparse.ArgumentParser(description="Prompt Loader")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)
# prompt 관련
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
parser.add_argument(
    "--num", type=int, default=1, help="Number of prompts to load from each category"
)
# model 관련
parser.add_argument(
    "--repo-id",
    type=str,
    default="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    help="Model repository ID",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda"
)
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["float16", "bfloat16", "float32"],
    help="Data type for the model",
)
# 출력 관련
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs",
    help="Directory to save generated images"
)

if __name__ == "__main__":
    args = parser.parse_args()
    logger = setup_logger(args.verbose)
    
    logger.debug("프롬프트 파일 로드 중: %s", args.prompt)
    prompt_dict = read_prompt_csv(args.prompt)
    
    logger.debug("디바이스 설정: %s", args.device)
    device = get_device(args.device)
    
    logger.debug("데이터 타입 설정: %s", args.dtype)
    dtype = get_dtype(args.dtype)
    
    # 요청된 카테고리들에서 프롬프트 가져오기
    selected_prompts = {}
    for category in args.category:
        if category in prompt_dict:
            # num 개수만큼만 가져오기 (카테고리에 있는 프롬프트 수보다 적은 경우)
            selected_prompts[category] = prompt_dict[category][
                : min(args.num, len(prompt_dict[category]))
            ]
            logger.debug("카테고리 '%s'에서 %d개의 프롬프트 로드됨", category, len(selected_prompts[category]))
        else:
            logger.warning("카테고리 '%s'가 프롬프트 파일에 없습니다", category)
    
    # 선택된 프롬프트 로깅
    for category, prompts in selected_prompts.items():
        logger.info("카테고리: %s", category)
        for i, prompt in enumerate(prompts, 1):
            logger.info(" %d. %s", i, prompt)
    
    # repo_id를 통해서 모델의 종류를 추론
    if args.repo_id in AVAILABLE_MODELS:
        repo_id = args.repo_id
        configs = AVAILABLE_MODELS[args.repo_id].copy()  # 복사본 생성
        model_type = configs.pop("type")  # type을 추출하고 configs에서 제거
        
        # 모델 로드
        if model_type == "sana":
            logger.debug("Sana 모델 로드 중: %s", repo_id)
            pipeline = get_sana(repo_id, device, dtype)
            logger.info("Sana 모델 로드 완료")
            
            # 출력 디렉토리 생성
            os.makedirs(args.output_dir, exist_ok=True)
            
            # 각 카테고리에 대하여 image 생성
            for category, prompts in selected_prompts.items():
                # 카테고리별 디렉토리 생성
                category_dir = os.path.join(args.output_dir, category)
                os.makedirs(category_dir, exist_ok=True)
                
                logger.info("카테고리 '%s'의 이미지 생성 시작", category)
                
                for i, prompt in enumerate(prompts):
                    logger.info("프롬프트 '%s'로 이미지 생성 중...", prompt)
                    
                    try:
                        # 이미지 생성
                        image = generate_image(pipeline, prompt, configs)
                        
                        # 이미지 저장
                        image_filename = f"{i+1}_{prompt[:30].replace(' ', '_')}.png"
                        image_path = os.path.join(category_dir, image_filename)
                        image.save(image_path)
                        
                        logger.info("이미지 저장 완료: %s", image_path)
                    except Exception as e:
                        logger.error("이미지 생성 중 오류 발생: %s", str(e))
                
                logger.info("카테고리 '%s'의 이미지 생성 완료", category)
        else:
            logger.error("지원하지 않는 모델 타입입니다: %s", model_type)
    else:
        logger.error("레포지토리 ID가 지원 목록에 없습니다: %s", args.repo_id)
    
    logger.info("모든 처리 완료")