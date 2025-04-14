import logging
from diffusers import DiffusionPipeline
from typing import Union, Dict
from PIL import Image


def generate_image(
    pipeline: DiffusionPipeline, prompt: str, configs: Dict[str, Union[str, int, float]]
) -> Image.Image:
    logger = logging.getLogger(__name__)

    image = pipeline(
        prompt=prompt,
        **configs,
    ).images[0]
    logger.debug("Image generated successfully.")
    return image
