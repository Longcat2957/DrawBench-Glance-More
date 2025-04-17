import logging
from diffusers import DiffusionPipeline
from typing import Union, Dict
from PIL import Image
import torch


def generate_image(
    pipeline: DiffusionPipeline,
    prompt: str,
    configs: Dict[str, Union[str, int, float]],
    generator: torch.Generator,
) -> Image.Image:
    logger = logging.getLogger(__name__)

    image = pipeline(
        prompt=prompt,
        generator=generator,
        **configs,
    ).images[0]
    logger.debug("Image generated successfully.")
    return image
