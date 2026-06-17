from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile

from pydantic import BaseModel

from file_reader import extract_text

from search_service import search_documents_hybrid
from openai_service import (
    get_document_embedding,
    analyze_document_info
)
from typing import Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from ocr_service import extract_ocr_texts
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentAnalyzeTextRequest(BaseModel):
    text: Optional[str] = None
    query: Optional[str] = None
    top: int = 5


def analyze_document_text(text: str, top: int):
    text = text.strip()

    if not text:
        raise ValueError("text must not be empty.")

    if top < 1 or top > 20:
        raise ValueError("top must be between 1 and 20.")

    query_vector = get_document_embedding(text)

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


def is_image_file(filename: str):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def extract_text_from_image_bytes(contents: bytes):
    ocr_items = extract_ocr_texts(contents)

    return "\n".join(
        item.get("text", "")
        for item in ocr_items
        if item.get("text", "").strip()
    )


@app.post("/documents/analyze/text")
def analyze_document_from_text(req: DocumentAnalyzeTextRequest):
    try:
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


@app.post("/documents/analyze/file")
async def analyze_document_from_file(
    file: UploadFile = File(...),
    top: int = Form(5)
):
    try:
        contents = await file.read()
        filename = file.filename or ""

        if is_image_file(filename):
            text = extract_text_from_image_bytes(contents)
            extract_method = "azure_vision_ocr"
        else:
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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        if "path" in locals() and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
