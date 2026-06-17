import os
import json
from functools import lru_cache

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


@lru_cache(maxsize=1)
def get_openai_client():
    return OpenAI(
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),
        base_url=get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    )


@lru_cache(maxsize=1)
def get_docs_openai_client():
    return OpenAI(
        api_key=get_required_env("AZURE_OPENAI_DOCS_API_KEY"),
        base_url=get_required_env("AZURE_OPENAI_DOCS_ENDPOINT").rstrip("/")
    )


def get_document_embedding(text):
    if not text:
        text = " "

    response = get_docs_openai_client().embeddings.create(
        model=get_required_env("AZURE_OPENAI_DOCS_DEPLOYMENT"),
        input=text
    )

    return response.data[0].embedding


def improve_translation(original_text, machine_translation, glossary_terms, target_lang):

    glossary_text = ""
    for t in glossary_terms:
        glossary_text += f"{t['ko']} = {t['en']}\n"

    prompt = f"""
You are a banking translation expert.

Target language: {target_lang}

Glossary:
{glossary_text}

Original:
{original_text}

Machine translation:
{machine_translation}

Return only final translation.
"""
    response = get_openai_client().chat.completions.create(
        model=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def analyze_document_info(query, retrieved_documents):
    context_lines = []

    for index, doc in enumerate(retrieved_documents, start=1):
        context_lines.append(
            f"""
[문서 {index}]
id: {doc.get("id", "")}
title: {doc.get("title", "")}
category: {doc.get("category", "")}
source_file: {doc.get("source_file", "")}
content: {doc.get("content", "")}
"""
        )

    context = "\n".join(context_lines)

    prompt = f"""
너는 금융 문서 분석 Agent다.
사용자가 입력한 원문을 기준으로 문서 정보를 분석한다.
Azure AI Search 검색 결과는 문서 유형 판단을 보조하는 참고자료일 뿐이다.

문서 유형은 반드시 아래 3개 중 하나만 선택한다.
- 금융 상품안내문
- 마케팅 콘텐츠
- 고객 안내서

분석 항목:
- document_type: 위 3개 중 하나
- document_character: 입력 원문에서 판단 가능한 문서 성격을 한 문장으로 요약
- included_information: 입력 원문에 실제로 포함된 주요 정보 목록
- key_numbers_preview: 입력 원문에 실제로 포함된 핵심 수치 목록
- legal_notice_detection.count: 입력 원문에서 법적 문구 또는 주의 문구로 볼 수 있는 항목 수
- legal_notice_detection.items: 입력 원문에서 감지된 법적/주의 문구 목록

절대 규칙:
- included_information에는 반드시 사용자 입력 원문에 실제로 있는 정보만 넣는다.
- key_numbers_preview에는 반드시 사용자 입력 원문에 실제로 있는 수치 정보만 넣는다.
- key_numbers_preview의 source_text에는 입력 원문에서 해당 수치를 확인할 수 있는 원문 일부를 그대로 넣는다.
- 기본금리, 기준금리, 우대금리, 가입기간, 예금자보호 한도, 대출한도, 한도, 수수료, 연회비, 만기, 중도해지 이율 등 금융 핵심 수치를 우선 추출한다.
- legal_notice_detection.items에는 반드시 사용자 입력 원문에 실제로 있는 문구만 넣는다.
- Azure AI Search 검색 결과에만 있는 금리, 가입기간, 우대조건, 중도해지 문구 등을 복사하지 않는다.
- 사용자 입력 원문에 없는 정보는 비슷해 보여도 절대 추가하지 않는다.
- 사용자 입력 원문이 "기준금리는 3.5%입니다."라면 included_information은 ["기준금리 3.5%"] 정도만 가능하고 legal_notice_detection.count는 0이어야 한다.
- 사용자 입력 원문이 "기준금리는 3.5%입니다."라면 key_numbers_preview는 [{{"label": "기준금리", "value": "3.5%", "source_text": "기준금리는 3.5%입니다."}}] 정도만 가능하다.
- 핵심 수치가 없으면 key_numbers_preview는 빈 배열로 둔다.
- 법적/주의 문구가 없으면 legal_notice_detection.count는 0, items는 빈 배열로 둔다.
- 반드시 JSON 객체만 응답한다. 설명 문장, markdown code block은 쓰지 않는다.

사용자 입력 원문:
{query}

Azure AI Search 검색 결과:
{context}

반환 형식:
{{
  "document_type": "금융 상품안내문",
  "document_character": "",
  "included_information": [],
  "key_numbers_preview": [
    {{
      "label": "기본금리",
      "value": "연 3.2%",
      "source_text": "기본금리 연 3.2%"
    }}
  ],
  "legal_notice_detection": {{
    "count": 0,
    "items": []
  }}
}}
"""

    try:
        response = get_openai_client().chat.completions.create(
            model=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
    except (BadRequestError, TypeError):
        response = get_openai_client().chat.completions.create(
            model=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
        result.setdefault("included_information", [])
        result.setdefault("key_numbers_preview", [])
        result.setdefault(
            "legal_notice_detection",
            {
                "count": 0,
                "items": []
            }
        )

        return result
    except json.JSONDecodeError:
        return {
            "document_type": "고객 안내서",
            "document_character": "분석 결과를 JSON으로 변환하지 못했습니다.",
            "included_information": [],
            "key_numbers_preview": [],
            "legal_notice_detection": {
                "count": 0,
                "items": []
            },
            "raw_response": content
        }
