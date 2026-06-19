import json
import re
from concurrent.futures import ThreadPoolExecutor

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
- title은 원문 제목이 있으면 번역하고, 명시 제목이 없으면 원문 근거 안에서 짧게 만든다.
- sections는 원문의 모든 문단·조항·항목을 빠짐없이 담은 "완전 번역"이어야 한다. 요약본이 아니다. 원문에 있는 내용은 길더라도 sections 어딘가에 모두 들어가야 한다.
- summary는 짧은 개요일 뿐이며, summary가 짧다는 이유로 sections에서 본문 내용을 생략하면 안 된다.
- full_text는 title, summary, sections를 사람이 읽기 좋은 순서로 합친 텍스트이며, 원문의 모든 정보를 포함해야 한다.
- 반드시 JSON 객체만 반환한다. markdown code block이나 설명 문장은 쓰지 않는다.

수치 보존 규칙 (가장 중요, 절대 위반 금지):
- 원문에 등장하는 모든 수치(금리, %, %p, 기간, 한도, 금액, 수수료, 날짜)는 translated_text에 하나도 빠짐없이 그대로 옮긴다. 값을 바꾸거나 반올림하지 않는다.
- 우대금리/우대조건처럼 여러 항목이 나열된 경우, 항목을 절대 합치거나 요약하거나 생략하지 말고 항목별로 각각의 값과 조건을 모두 번역해 넣는다.
  예) "급여이체 0.50%, 대환신청 2.00%, MGM 1.50%"가 원문에 있으면 번역문에도 세 항목과 각 수치가 모두 있어야 한다. "우대금리 제공"처럼 뭉뚱그리면 안 된다.
- 각 우대조건은 "조건명 + 해당 우대율(또는 한도)" 쌍을 유지한다. 조건명만 적고 수치를 빠뜨리거나, 수치만 적고 조건을 빠뜨리지 않는다.
- 둘 이상의 조건이 같은 우대율을 갖더라도 절대 하나로 합치지 않는다. 값이 같다는 이유로 중복을 제거하거나 "여러 조건 0.50%"처럼 묶지 말고, 조건마다 그 조건의 우대율을 1:1로 따로 나열한다.
  예) 원문에 "급여 입금 0.50%, 외환 환전·송금 실적 0.50%"가 있으면, 번역문에도 "급여 입금: 연 0.50%"와 "외환 환전·송금 실적: 연 0.50%" 두 항목이 각각 따로 있어야 한다. "preferential rate up to 0.50%"처럼 한 줄로 합치면 안 된다.
- summary는 1~2문장 요약이되, 수치 나열 자체는 summary가 아니라 sections에 항목별로 빠짐없이 담는다. (summary가 짧다는 이유로 수치를 버리지 않는다)
- 번역을 끝내기 전에 원문의 모든 숫자/퍼센트/기간/한도/수수료가 translated_text 어딘가에 실제로 존재하는지 스스로 점검하고, 누락된 값이 있으면 반드시 보완한 뒤 응답한다.
- 원문에 없는 상품명, 조건, 혜택, 수치, 법적 고지는 새로 추가하지 않는다. (보존은 하되 창작은 금지)

법적·주의 문구 보존 규칙 (수치만큼 중요, 절대 위반 금지):
- 유의사항, 경고, 예외 규정, 면책, 불이익 가능성 등 법적/주의 문장은 한 줄로 축약하지 말고, 조건과 결과(예: "~한 경우 ~될 수 있음")를 모두 살려 완전하게 번역한다.
  예) "재직상태·신용상태·은행 심사기준에 따라 연장되지 않을 수 있고 거래조건이 변경될 수 있으니 유의" → 어느 기준에 따라 / 무엇이 일어날 수 있는지를 모두 번역에 남긴다. "subject to the bank's review"처럼 뭉뚱그리면 안 된다.
- 예외·단서 조항(예: "5천만원 미만인 경우 ~에는 연체이자율이 적용되지 않습니다")은 적용 대상·조건·예외 내용을 그대로 옮기고 통째로 생략하지 않는다.
- 법령·규정의 정식 명칭(예: 「인지세법」)은 반드시 번역문에 명시한다. 법령명을 빼고 일반 설명으로 대체하지 않는다.
- "가능 여부: 부/가", "해당/비해당", "지원/미지원" 같은 가부(可否)·유무 표기는 그 값(부=불가/불가능 등)을 명확히 번역에 반영한다. 항목만 적고 가부 값을 빠뜨리지 않는다.
- 번역을 끝내기 전에 원문의 모든 유의사항·예외규정·법령명·가부 표기가 translated_text에 실제로 반영됐는지 스스로 점검하고, 누락분이 있으면 보완한 뒤 응답한다.

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


def translate_single_language(
    language,
    source_text,
    document_analysis,
    tone_style,
    finance_terms
):
    # 언어 1개만 번역한다. 여러 언어를 한 호출에 넣으면 뒤쪽 언어가 압축·누락되므로,
    # 언어별로 독립 호출해 각 언어가 프롬프트 규칙(수치/법적문구 보존)을 온전히 받게 한다.
    prompt = build_translation_prompt(
        source_text=source_text,
        document_analysis=document_analysis or {},
        target_languages=[language],
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

    # 단일 언어 호출이므로 해당 언어 항목을 찾고, 못 찾으면 첫 항목으로 폴백한다.
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
        raw_translations[0] if raw_translations else {}
    )

    return normalize_translation(language, matching_translation)


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

    # 언어별로 독립 호출하되, 동기 클라이언트를 스레드풀로 병렬 실행해 전체 소요 시간을
    # 단일 언어 번역 수준으로 유지한다. 결과는 입력 언어 순서를 그대로 보존한다.
    with ThreadPoolExecutor(max_workers=min(len(target_languages), 4)) as executor:
        translations = list(
            executor.map(
                lambda language: translate_single_language(
                    language,
                    source_text,
                    document_analysis,
                    tone_style,
                    finance_terms
                ),
                target_languages
            )
        )

    return {
        "translations": translations
    }
