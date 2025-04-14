import os
import pandas as pd
from typing import List, Dict, Any


def read_prompt_csv(p: str) -> Dict[str, List[str]]:
    """
    DrawBench Sample
    ```
        Prompts,Category
        A red colored car.,Colors
        A black colored car.,Colors
        A pink colored car.,Colors
        A black colored dog.,Colors
        A red colored dog.,Colors
        A blue colored dog.,Colors
    ```
    Estimated result
    ```
    {
        "Colors": [
            "A red colored car.",
            "A black colored car.",
            "A pink colored car.",
            "A black colored dog.",
            "A red colored dog.",
            "A blue colored dog."
        ]
    }
    ```
    """
    if not os.path.exists(p):
        raise FileNotFoundError(f"File {p} does not exist.")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # CSV 파일에 필요한 열이 있는지 확인
    if "Prompts" not in df.columns or "Category" not in df.columns:
        raise ValueError("CSV file must contain 'Prompts' and 'Category' columns")

    # 카테고리별로 프롬프트를 정리
    result = {}
    for _, row in df.iterrows():
        category = row["Category"]
        prompt = row["Prompts"]

        if category not in result:
            result[category] = []

        result[category].append(prompt)

    return result
