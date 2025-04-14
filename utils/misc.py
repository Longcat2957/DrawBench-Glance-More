import torch


def get_device(device: str = "cpu") -> torch.device:
    """현재 사용 가능한 GPU 또는 CPU 장치를 반환합니다."""
    device = device.lower()

    # 1. CPU 명시적 지정
    if device == "cpu":
        return torch.device("cpu")

    # 2. CUDA 사용 가능한지 먼저 검사
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # 3. cuda:0, cuda:1 등 지정된 경우
    if ":" in device:
        try:
            index = int(device.split(":")[1])
            if index < torch.cuda.device_count():
                return torch.device(device)
            else:
                raise ValueError(f"GPU index {index} is out of range")
        except (ValueError, IndexError) as e:
            return torch.device("cuda")

    # 4. 단순 'cuda'인 경우
    return torch.device("cuda")

def get_dtype(dtype: str = "float16") -> torch.dtype:
    """사용자 지정 데이터 유형을 반환합니다."""
    dtype = dtype.lower()

    if dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported data type: {dtype}")