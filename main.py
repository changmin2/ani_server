from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from translator import translate
from file_reader import extract_text

from search_service import search_finance_terms
from openai_service import improve_translation
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str
    target_languages: List[str]
    source: str


@app.post("/translate/text")
def translate_text(req: TranslateRequest):
    try:
        translations = {}

        glossary_terms = search_finance_terms(req.text)

        for lang in req.target_languages:
            machine_translation = translate(req.text, lang)

            final_translation = improve_translation(
                req.text,
                machine_translation,
                glossary_terms,
                lang
            )

            translations[lang] = {
                "machine_translation": machine_translation,
                "glossary": glossary_terms,
                "final_translation": final_translation
            }

        return {
            "translations": translations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }


@app.post("/translate/file")
async def translate_file(
    file: UploadFile = File(...),
    target_languages: List[str] = Form(...),
    source: str = Form(...),
):
    try:
        contents = await file.read()
        path = f"temp_{file.filename}"

        with open(path, "wb") as f:
            f.write(contents)

        # 파일 텍스트 추출
        text = extract_text(path)

        # 금융 용어 검색은 원문 기준이라 한 번만
        glossary_terms = search_finance_terms(text)

        translations = {}

        for lang in target_languages:
            # 1차 번역
            machine_translation = translate(text, lang)

            # GPT 보정
            final_translation = improve_translation(
                text,
                machine_translation,
                glossary_terms,
                lang
            )

            translations[lang] = {
                "machine_translation": machine_translation,
                "glossary": glossary_terms,
                "final_translation": final_translation,
            }

        return {
            "file_name": file.filename,
            "source": source,
            "original_text": text,
            "translations": translations,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }

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