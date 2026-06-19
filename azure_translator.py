import os

import requests
from dotenv import load_dotenv

load_dotenv()

# Azure Translator REST(v3.0)는 짧은 텍스트를 배열로 한 번에 번역할 수 있어
# PDF span 단위(줄/조각) 번역에 적합하다. 한 요청당 배열 1,000개 / 50,000자 제한이 있어
# 안전하게 100개씩 끊어 보낸다.
_BATCH_SIZE = 100


def _get_config():
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    region = os.getenv("AZURE_TRANSLATOR_REGION")

    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_TRANSLATOR_ENDPOINT 또는 AZURE_TRANSLATOR_KEY가 .env에 없습니다."
        )

    return endpoint.rstrip("/"), key, region


def translate_texts(texts, target_code, source_code=None):
    """여러 짧은 텍스트를 한꺼번에 번역한다.

    입력 순서와 동일한 순서의 번역 결과 리스트를 돌려준다.
    공백/빈 문자열은 번역 요청에서 제외하고 원본을 그대로 돌려준다.
    """
    endpoint, key, region = _get_config()

    # 빈 칸은 Azure가 에러를 내므로 인덱스를 기억해 두고 번역 대상에서 뺀다.
    index_map = [index for index, text in enumerate(texts) if text and text.strip()]
    results = list(texts)

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }

    if region:
        headers["Ocp-Apim-Subscription-Region"] = region

    params = {"api-version": "3.0", "to": target_code}

    if source_code:
        params["from"] = source_code

    url = f"{endpoint}/translate"

    for start in range(0, len(index_map), _BATCH_SIZE):
        chunk_indices = index_map[start:start + _BATCH_SIZE]
        body = [{"Text": texts[index]} for index in chunk_indices]

        response = requests.post(url, params=params, headers=headers, json=body, timeout=60)
        response.raise_for_status()

        for index, item in zip(chunk_indices, response.json()):
            translations = item.get("translations") or []
            if translations:
                results[index] = translations[0].get("text", texts[index])

    return results
