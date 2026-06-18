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
    # 문서 분석 Agent가 사용하는 Azure OpenAI chat 모델 클라이언트.
    return OpenAI(
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),
        base_url=get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    )


@lru_cache(maxsize=1)
def get_docs_openai_client():
    # 문서 검색용 임베딩 모델 클라이언트.
    # 업로드 스크립트(upload_docs.py)와 같은 임베딩 배포를 사용해야 검색 벡터 차원이 맞는다.
    return OpenAI(
        api_key=get_required_env("AZURE_OPENAI_DOCS_API_KEY"),
        base_url=get_required_env("AZURE_OPENAI_DOCS_ENDPOINT").rstrip("/")
    )


def get_document_embedding(text):
    # AI Search 벡터 검색에 사용할 입력 원문 임베딩을 만든다.
    # 빈 문자열은 임베딩 API 에러를 피하기 위해 공백 하나로 대체한다.
    if not text:
        text = " "

    response = get_docs_openai_client().embeddings.create(
        model=get_required_env("AZURE_OPENAI_DOCS_DEPLOYMENT"),
        input=text
    )

    return response.data[0].embedding


def calculate_document_confidence(text, analysis, retrieved_documents):
    # 화면의 "신뢰도"에 표시할 점수를 서버에서 계산한다.
    # 모델에게 자체 평가를 맡기면 일관성이 떨어질 수 있어, 추출량/검색 근거/문서 길이 기반으로 산정한다.
    score = 0.35
    text_length = len(text.strip())

    # 원문 길이가 너무 짧으면 판단 근거가 부족하므로, 일정 길이 이상일 때만 가점을 준다.
    if text_length >= 50:
        score += 0.08
    if text_length >= 150:
        score += 0.07

    included_information = analysis.get("included_information", [])
    key_numbers_preview = analysis.get("key_numbers_preview", [])
    legal_notice_detection = analysis.get("legal_notice_detection", {})

    # 실제 원문에서 뽑힌 정보가 많을수록 분석 근거가 풍부하다고 본다.
    score += min(len(included_information) * 0.04, 0.16)
    score += min(len(key_numbers_preview) * 0.05, 0.20)

    if legal_notice_detection.get("count", 0) > 0:
        score += 0.04

    if retrieved_documents:
        score += 0.08

        # 상위 검색 결과의 category가 일관되면 문서 유형 판단도 더 안정적이라고 본다.
        categories = [
            doc.get("category", "")
            for doc in retrieved_documents[:3]
            if doc.get("category")
        ]

        if categories:
            most_common_count = max(categories.count(category) for category in set(categories))
            score += (most_common_count / len(categories)) * 0.07

    return round(min(score, 0.95), 2)


def analyze_document_info(query, retrieved_documents):
    # LLM에게 넘길 AI Search 검색 결과를 사람이 읽기 쉬운 컨텍스트로 직렬화한다.
    # 중요한 점: 이 컨텍스트는 "참고자료"일 뿐이고, 실제 추출값은 사용자 입력 원문에서만 뽑도록 프롬프트에서 강제한다.
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
- document_structure.title: 입력 원문에서 확인 가능한 제목 또는 가장 적절한 문서명
- document_structure.body_sections: 입력 원문에서 확인 가능한 본문 영역/구성 요소 목록
- document_structure.notice_phrases: 입력 원문에서 확인 가능한 주의 문구 목록
- included_information: 입력 원문에 실제로 포함된 주요 정보 목록
- key_numbers_preview: 입력 원문에 실제로 포함된 핵심 수치 목록
- legal_notice_detection.count: 입력 원문에서 법적 문구 또는 주의 문구로 볼 수 있는 항목 수
- legal_notice_detection.items: 입력 원문에서 감지된 법적/주의 문구 목록

절대 규칙:
- included_information에는 반드시 사용자 입력 원문에 실제로 있는 정보만 넣는다.
- document_structure는 반드시 사용자 입력 원문에서 판단 가능한 내용만 넣는다.
- 제목이 명시되어 있지 않으면 입력 원문 내용을 바탕으로 짧은 문서명을 만들되, 추측성 상품명은 만들지 않는다.
- body_sections에는 상품 개요, 가입 대상, 가입기간, 금리, 우대조건, 해지조건, 이용방법 등 원문에 실제로 드러나는 영역만 넣는다.
- notice_phrases에는 예금자보호, 중도해지, 원금손실, 환율변동, 수수료, 이용제한 등 주의/고지 문구가 실제로 있을 때만 넣는다.
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
  "document_structure": {{
    "title": "",
    "body_sections": [],
    "notice_phrases": []
  }},
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
        # JSON 모드를 지원하는 Azure OpenAI 배포에서는 응답 형식을 JSON 객체로 강제한다.
        response = get_openai_client().chat.completions.create(
            model=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
    except (BadRequestError, TypeError):
        # 일부 Azure 배포/SDK 조합은 response_format을 지원하지 않을 수 있으므로 일반 호출로 fallback한다.
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
        # 프론트가 항상 같은 shape을 기대할 수 있도록 누락된 필드는 기본값으로 보정한다.
        result.setdefault(
            "document_structure",
            {
                "title": "",
                "body_sections": [],
                "notice_phrases": []
            }
        )
        result.setdefault("included_information", [])
        result.setdefault("key_numbers_preview", [])
        result.setdefault(
            "legal_notice_detection",
            {
                "count": 0,
                "items": []
            }
        )
        result["confidence"] = calculate_document_confidence(
            query,
            result,
            retrieved_documents
        )

        return result
    except json.JSONDecodeError:
        # 모델이 JSON이 아닌 답변을 반환한 경우에도 프론트가 깨지지 않도록 fallback JSON을 내려준다.
        result = {
            "document_type": "고객 안내서",
            "document_character": "분석 결과를 JSON으로 변환하지 못했습니다.",
            "document_structure": {
                "title": "",
                "body_sections": [],
                "notice_phrases": []
            },
            "included_information": [],
            "key_numbers_preview": [],
            "legal_notice_detection": {
                "count": 0,
                "items": []
            },
            "raw_response": content
        }
        result["confidence"] = calculate_document_confidence(
            query,
            result,
            retrieved_documents
        )

        return result
