from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import tempfile

from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from cache_service import clear_cache, get_or_set_cache, make_cache_key, normalize_text
from file_reader import extract_text

from search_service import (
    find_finance_terms_in_text,
    search_documents_hybrid,
    search_finance_terms_hybrid
)
from openai_service import (
    get_document_embedding,
    analyze_document_info
)
from translation_service import translate_document, get_language_code
from validation_agent import call_gpt_41_model, call_validation_agent
from layout_translation_service import translate_pdf_layout, render_translated_page_png
from typing import Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from ocr_service import extract_ocr_texts
app = FastAPI()

# 프론트 개발 서버(Vite 기본 포트)에서 백엔드 API를 호출할 수 있도록 CORS를 허용한다.
# 여러 작업자가 같은 네트워크에서 Vite dev server로 접속할 수 있어 localhost뿐 아니라
# 5173 포트로 들어오는 개발 origin을 허용한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_origin_regex=r"^http://[^/]+:517[0-9]$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentAnalyzeTextRequest(BaseModel):
    # 신규 화면에서는 text를 사용하지만, 이전 테스트 코드 호환을 위해 query도 허용한다.
    text: Optional[str] = None
    query: Optional[str] = None
    # Azure AI Search에서 가져올 관련 문서 수. 너무 크면 응답/프롬프트가 불필요하게 커진다.
    top: int = 5


class FinanceTermSearchRequest(BaseModel):
    text: str
    top: int = 12


class DocumentTranslateRequest(BaseModel):
    source_text: str
    document_analysis: dict = Field(default_factory=dict)
    target_languages: list[str]
    tone_style: str = "공식적이고 신뢰감 있는 금융 문체"
    finance_terms: list[dict] = Field(default_factory=list)


class ValidationKeyInformation(BaseModel):
    label: str
    sourceValue: str


class ValidationTranslationInput(BaseModel):
    targetLanguage: str
    translatedText: str


class ValidationAgentRequest(BaseModel):
    documentType: str = "금융상품 안내문"
    sourceText: str = "우대금리는 최대 연 0.50%p 제공됩니다."
    translations: list[ValidationTranslationInput] = Field(default_factory=list)
    keyInformation: list[ValidationKeyInformation] = Field(
        default_factory=lambda: [
            ValidationKeyInformation(
                label="우대금리",
                sourceValue="최대 연 0.50%p"
            )
        ]
    )


def analyze_document_text(text: str, top: int):
    # 텍스트 입력과 파일 OCR/텍스트 추출 결과가 공통으로 타는 핵심 분석 파이프라인이다.
    # 1) 입력 원문 임베딩 생성
    # 2) AI Search에서 키워드 + 벡터 하이브리드 검색
    # 3) 검색 결과를 참고 자료로 넣어 Azure OpenAI Agent가 JSON 분석 결과 생성
    text = text.strip()

    if not text:
        raise ValueError("text must not be empty.")

    if top < 1 or top > 20:
        raise ValueError("top must be between 1 and 20.")

    cache_key = make_cache_key(
        "document_analysis",
        {
            "text": normalize_text(text),
            "top": top
        }
    )

    def run_analysis():
        query_vector = get_document_embedding(text)

        # 검색 결과는 최종 추출값의 원천이 아니라, 문서 유형 판단을 보조하는 근거 자료로만 사용한다.
        retrieved_documents = search_documents_hybrid(
            query=text,
            query_vector=query_vector,
            top=top
        )

        analysis = analyze_document_info(
            query=text,
            retrieved_documents=retrieved_documents
        )

        return {
            "text": text,
            "analysis": analysis,
            "retrieved_documents": retrieved_documents
        }

    result, cache_hit = get_or_set_cache(cache_key, run_analysis)
    result["cache_hit"] = cache_hit

    return result


def search_finance_term_matches(text: str, top: int):
    text = text.strip()

    if not text:
        raise ValueError("text must not be empty.")

    if top < 1 or top > 50:
        raise ValueError("top must be between 1 and 50.")

    cache_key = make_cache_key(
        "finance_terms",
        {
            "text": normalize_text(text),
            "top": top
        }
    )

    def run_search():
        matches = find_finance_terms_in_text(text, top=top)

        if not matches:
            query_vector = get_document_embedding(text)
            matches = search_finance_terms_hybrid(
                query=text,
                query_vector=query_vector,
                top=top
            )

        return {
            "text": text,
            "matches": matches
        }

    result, cache_hit = get_or_set_cache(cache_key, run_search)
    result["cache_hit"] = cache_hit

    return result


def is_image_file(filename: str):
    # 이미지 파일은 일반 파일 텍스트 추출 대신 Azure Vision OCR 경로를 탄다.
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def extract_text_from_image_bytes(contents: bytes):
    # Azure Vision OCR은 줄 단위 결과와 좌표를 함께 주므로, 분석 API에는 텍스트만 합쳐서 넘긴다.
    ocr_items = extract_ocr_texts(contents)

    return "\n".join(
        item.get("text", "")
        for item in ocr_items
        if item.get("text", "").strip()
    )


def process_uploaded_document(contents: bytes, filename: str, top: int):
    path = None

    try:
        if is_image_file(filename):
            # 이미지 업로드: Azure Vision OCR로 글자를 먼저 뽑은 뒤 분석한다.
            text = extract_text_from_image_bytes(contents)
            extract_method = "azure_vision_ocr"
        else:
            # PDF/DOCX/TXT 업로드: 임시 파일로 저장한 뒤 기존 extract_text 유틸로 본문을 추출한다.
            _, suffix = os.path.splitext(filename)

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                path = tmp.name
                tmp.write(contents)

            text = extract_text(path)
            extract_method = "file_text_extraction"

        if not text.strip():
            raise ValueError("file text could not be extracted.")

        result = analyze_document_text(text, top)

        return {
            "file_name": filename,
            "extract_method": extract_method,
            **result
        }

    finally:
        # PDF/DOCX/TXT 경로에서 만든 임시 파일은 요청 처리 후 반드시 삭제한다.
        if path and os.path.exists(path):
            os.remove(path)


@app.post("/documents/analyze/text")
def analyze_document_from_text(req: DocumentAnalyzeTextRequest):
    try:
        # 프론트는 text로 보내고, 과거 테스트용 요청은 query로 들어올 수 있다.
        text = req.text if req.text is not None else req.query

        if text is None:
            raise ValueError("text must not be empty.")

        return analyze_document_text(text, req.top)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/finance-terms/search")
def search_finance_terms(req: FinanceTermSearchRequest):
    try:
        return search_finance_term_matches(req.text, req.top)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents/translate")
def translate_document_from_analysis(req: DocumentTranslateRequest):
    try:
        source_text = req.source_text.strip()
        cache_key = make_cache_key(
            "document_translation",
            {
                "source_text": normalize_text(source_text),
                "document_analysis": req.document_analysis,
                "target_languages": req.target_languages,
                "tone_style": req.tone_style,
                "finance_terms": req.finance_terms
            }
        )

        def run_translation():
            return translate_document(
                source_text=source_text,
                document_analysis=req.document_analysis,
                target_languages=req.target_languages,
                tone_style=req.tone_style,
                finance_terms=req.finance_terms
            )

        result, cache_hit = get_or_set_cache(cache_key, run_translation)
        result["cache_hit"] = cache_hit

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents/validation")
async def validate_translation_with_agent(req: ValidationAgentRequest):
    try:
        source_text = req.sourceText.strip()
        validation_items = []

        for item in req.translations:
            target_language = get_language_code(item.targetLanguage.strip())
            translated_text = item.translatedText.strip()

            if not target_language or not translated_text:
                continue

            if any(
                existing["targetLanguage"] == target_language
                for existing in validation_items
            ):
                continue

            validation_items.append({
                "targetLanguage": target_language,
                "translatedText": translated_text,
                "translation": None,
            })

        if not source_text:
            raise ValueError("sourceText must not be empty.")

        if not validation_items:
            raise ValueError("translations must not be empty.")

        if len(validation_items) > 4:
            raise ValueError("translations must include at most 4 languages.")

        results = []

        for item in validation_items:
            validation_payload = {
                "documentType": req.documentType,
                "targetLanguage": item["targetLanguage"],
                "sourceText": source_text,
                "translatedText": item["translatedText"],
                "keyInformation": [
                    item.model_dump()
                    for item in req.keyInformation
                ],
            }

            validation_result = await run_in_threadpool(
                call_validation_agent,
                validation_payload,
            )

            if isinstance(validation_result, dict) and validation_result.get("skipped"):
                validation_result = await run_in_threadpool(
                    call_gpt_41_model,
                    validation_payload,
                )

            results.append({
                "targetLanguage": item["targetLanguage"],
                "translatedText": item["translatedText"],
                "validationResult": validation_result,
            })

        return {"results": results}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents/analyze/file")
async def analyze_document_from_file(
    file: UploadFile = File(...),
    top: int = Form(5)
):
    try:
        contents = await file.read()
        filename = file.filename or ""

        return await run_in_threadpool(
            process_uploaded_document,
            contents,
            filename,
            top
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents/translate-layout")
async def translate_document_layout(
    file: UploadFile = File(...),
    target_language: str = Form("en"),
):
    """[프로토타입] 텍스트형 PDF의 디자인/위치를 유지한 채 언어만 바꿔 PDF로 돌려준다."""
    try:
        filename = file.filename or ""

        if not filename.lower().endswith(".pdf"):
            raise ValueError("현재 프로토타입은 PDF 파일만 지원합니다.")

        contents = await file.read()
        target_code = get_language_code(target_language)

        translated_pdf = await run_in_threadpool(
            translate_pdf_layout,
            contents,
            target_code,
        )

        download_name = f"translated_{target_code}.pdf"

        return Response(
            content=translated_pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/documents/translate-layout/preview")
async def translate_document_layout_preview(
    file: UploadFile = File(...),
    target_language: str = Form("en"),
    page: int = Form(0),
):
    """[프로토타입] PDF 첫 페이지를 번역해 PNG 이미지로 돌려준다. (5페이지 미리보기용)"""
    try:
        filename = file.filename or ""

        if not filename.lower().endswith(".pdf"):
            raise ValueError("현재 미리보기는 PDF 파일만 지원합니다.")

        contents = await file.read()
        target_code = get_language_code(target_language)

        png_bytes = await run_in_threadpool(
            render_translated_page_png,
            contents,
            target_code,
            page,
        )

        return Response(content=png_bytes, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/cache/clear")
def clear_server_cache():
    # 서버 재시작 없이 메모리 캐시(동일 요청 재사용분)를 비운다.
    # 프롬프트/로직 수정 후 같은 원문으로 다시 테스트할 때 사용한다.
    cleared = clear_cache()

    return {"cleared": cleared}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
