"""
NVIDIA SANA

Model List:
    SANA-1.5
        SANA1.5_4.8B_1024px_diffusers (Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers) [https://huggingface.co/Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers]
        SANA1,5_1,6B_1024px_diffusers (Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers) [https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers]

    SANA-SPRINT
        Sana_Sprint_0.6B_1024px_diffusers (Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) [https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers]
        Sana_Sprint_1.6B_1024px_diffusers (Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) [https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers]

"""

import torch
from diffusers import SanaPipeline, SanaSprintPipeline
from typing import Union


def get_sana(
    repo_id: str, device: str, dtype: torch.dtype
) -> Union[SanaPipeline, SanaSprintPipeline]:
    """
    Load SanaPipeline or SanaSprintPipeline from Hugging Face Hub.

    """

    if "Sana_Sprint" in repo_id:
        pipeline = SanaSprintPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
        )
    else:
        pipeline = SanaPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
        )

    pipeline.to(device)
    return pipeline
