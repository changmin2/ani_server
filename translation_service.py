import json
import re

from openai import BadRequestError

from openai_service import get_openai_client, get_required_env


LANGUAGE_CODE_MAP = {
    "en": "en",
    "영어": "en",
    "english": "en",
    "zh": "zh",
    "중국어": "zh",
    "chinese": "zh",
    "中文": "zh",
    "vi": "vi",
    "베트남어": "vi",
    "vietnamese": "vi",
    "tiếng việt": "vi",
    "ja": "ja",
    "jp": "ja",
    "일본어": "ja",
    "japanese": "ja",
    "kk": "kk",
    "카자흐스탄어": "kk",
    "kazakh": "kk",
    "қазақша": "kk",
}


def get_language_code(language: str):
    normalized = language.casefold()

    for keyword, code in LANGUAGE_CODE_MAP.items():
        if keyword.casefold() in normalized:
            return code

    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "unknown"


def normalize_sections(sections):
    if not isinstance(sections, list):
        return []

    normalized_sections = []

    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            continue

        try:
            order = int(section.get("order") or index)
        except (TypeError, ValueError):
            order = index

        normalized_sections.append({
            "id": str(section.get("id") or f"section_{index}"),
            "order": order,
            "source_label": str(section.get("source_label") or ""),
            "source_text": str(section.get("source_text") or ""),
            "translated_label": str(section.get("translated_label") or ""),
            "translated_text": str(section.get("translated_text") or ""),
        })

    return normalized_sections


def normalize_translation(language, item):
    if not isinstance(item, dict):
        item = {}

    title = str(item.get("title") or "")
    summary = str(item.get("summary") or "")
    sections = normalize_sections(item.get("sections", []))
    full_text = str(item.get("full_text") or "").strip()

    if not full_text:
        section_texts = [
            f"{section['translated_label']}\n{section['translated_text']}".strip()
            for section in sections
            if section["translated_label"] or section["translated_text"]
        ]
        full_text = "\n\n".join(
            part
            for part in [title, summary, *section_texts]
            if part
        )

    return {
        "language": str(item.get("language") or language),
        "language_code": str(item.get("language_code") or get_language_code(language)),
        "title": title,
        "summary": summary,
        "sections": sections,
        "full_text": full_text,
    }


def build_terms_context(finance_terms):
    if not finance_terms:
        return "제공된 금융 용어집 없음"

    lines = []

    for term in finance_terms[:80]:
        if not isinstance(term, dict):
            continue

        lines.append(
            json.dumps(
                {
                    "term_ko": term.get("term_ko", ""),
                    "term_en": term.get("term_en", ""),
                    "term_vi": term.get("term_vi", ""),
                    "term_zh": term.get("term_zh", ""),
                    "term_kk": term.get("term_kk", ""),
                    "description": term.get("description", ""),
                },
                ensure_ascii=False
            )
        )

    return "\n".join(lines) if lines else "제공된 금융 용어집 없음"


def build_translation_prompt(
    source_text,
    document_analysis,
    target_languages,
    tone_style,
    finance_terms
):
    return f"""
너는 금융 문서 전문 번역 Agent다.
사용자 원문을 대상 언어별로 번역하되, 화면 검토와 후속 검수에 바로 쓸 수 있는 구조화 JSON을 만든다.

번역 방식:
- 줄 단위로 기계적으로 쪼개지 말고, 제목/요약/섹션 단위로 문서 구조를 만든다.
- 각 섹션은 source_label, source_text, translated_label, translated_text를 포함한다.
- source_label과 source_text는 반드시 사용자 원문에서 확인 가능한 내용만 사용한다.
- translated_label과 translated_text는 해당 대상 언어로 자연스럽게 번역한다.
- 금융 용어집에 있는 용어는 대상 언어별 번역어를 우선 사용한다.
- 핵심 수치, 금리, 기간, 한도, 수수료, 날짜, 법적/주의 문구의 의미와 값을 바꾸지 않는다.
- 원문에 없는 상품명, 조건, 혜택, 수치, 법적 고지는 추가하지 않는다.
- title은 원문 제목이 있으면 번역하고, 명시 제목이 없으면 원문 근거 안에서 짧게 만든다.
- summary는 원문 내용을 벗어나지 않는 1~2문장 요약 번역이다.
- full_text는 title, summary, sections를 사람이 읽기 좋은 순서로 합친 텍스트다.
- 반드시 JSON 객체만 반환한다. markdown code block이나 설명 문장은 쓰지 않는다.

대상 언어:
{json.dumps(target_languages, ensure_ascii=False)}

톤/스타일:
{tone_style}

문서 분석 결과:
{json.dumps(document_analysis, ensure_ascii=False)}

금융 용어집:
{build_terms_context(finance_terms)}

사용자 원문:
{source_text}

반환 형식:
{{
  "translations": [
    {{
      "language": "영어 (English)",
      "language_code": "en",
      "title": "",
      "summary": "",
      "sections": [
        {{
          "id": "section_1",
          "order": 1,
          "source_label": "",
          "source_text": "",
          "translated_label": "",
          "translated_text": ""
        }}
      ],
      "full_text": ""
    }}
  ]
}}
"""


def translate_document(
    source_text,
    document_analysis,
    target_languages,
    tone_style,
    finance_terms=None
):
    source_text = source_text.strip()

    if not source_text:
        raise ValueError("source_text must not be empty.")

    if not target_languages:
        raise ValueError("target_languages must not be empty.")

    prompt = build_translation_prompt(
        source_text=source_text,
        document_analysis=document_analysis or {},
        target_languages=target_languages,
        tone_style=tone_style or "공식적이고 신뢰감 있는 금융 문체",
        finance_terms=finance_terms or []
    )

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
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("translation response was not valid JSON.") from exc

    raw_translations = parsed.get("translations", [])

    if not isinstance(raw_translations, list):
        raise ValueError("translations must be a list.")

    translations = []

    for index, language in enumerate(target_languages):
        matching_translation = next(
            (
                item
                for item in raw_translations
                if isinstance(item, dict)
                and (
                    item.get("language") == language
                    or item.get("language_code") == get_language_code(language)
                )
            ),
            raw_translations[index] if index < len(raw_translations) else {}
        )

        translations.append(normalize_translation(language, matching_translation))

    return {
        "translations": translations
    }
