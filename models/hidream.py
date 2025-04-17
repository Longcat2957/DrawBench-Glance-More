import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import UniPCMultistepScheduler, HiDreamImagePipeline


def get_hidream(repo_id: str, device: str, dtype: torch.dtype, shift: float = 3.0):
    print(f"!!! dtype: {dtype}")
    scheduler = UniPCMultistepScheduler(
        flow_shift=shift, prediction_type="flow_prediction", use_flow_sigmas=True
    )

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False
    )

    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )
    pipe = HiDreamImagePipeline.from_pretrained(
        repo_id,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe
