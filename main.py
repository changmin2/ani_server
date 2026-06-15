from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

from pydantic import BaseModel

from translator import translate
from file_reader import extract_text

from search_service import search_finance_terms
from openai_service import improve_translation
from pydantic import BaseModel
from typing import List, Dict, Any
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



def make_translated_image(contents: bytes, texts: list, lang: str):
    image = Image.open(BytesIO(contents)).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("Arial Unicode.ttf", 18)
    except:
        font = ImageFont.load_default()

    for item in texts:
        translated = item.get("translations", {}).get(lang, "")
        if not translated:
            continue

        x = item.get("x", 0)
        y = item.get("y", 0)
        w = item.get("width", 120)
        h = item.get("height", 24)

        draw.rectangle(
            [x, y, x + w, y + h],
            fill="white"
        )

        draw.text(
            (x, y),
            translated,
            fill="black",
            font=font
        )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    target_languages: str = Form(...)
):
    try:
        contents = await file.read()

        texts = extract_ocr_texts(contents)

        languages = [
            lang.strip()
            for lang in target_languages.split(",")
            if lang.strip()
        ]

        translated_texts = []

        for item in texts:
            original_text = item.get("text", "")
            translations = {}

            for lang in languages:
                if original_text.strip():
                    translated = translate(original_text, lang)
                else:
                    translated = ""

                translations[lang] = translated

            translated_texts.append({
                **item,
                "translations": translations
            })

        translated_images = {}

        for lang in languages:
            translated_images[lang] = make_translated_image(
                contents,
                translated_texts,
                lang
            )

        return {
            "file_name": file.filename,
            "texts": translated_texts,
            "translated_images": translated_images
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )