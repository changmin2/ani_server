from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile

from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from cache_service import get_or_set_cache, make_cache_key, normalize_text
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
from translation_service import translate_document
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
