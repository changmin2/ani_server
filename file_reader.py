from docx import Document
import fitz
import pytesseract
from PIL import Image


def extract_text(file_path):

    lower_path = file_path.lower()

    # TXT
    if lower_path.endswith(".txt"):

        encodings = [
            "utf-8",
            "utf-16",
            "cp949",
            "euc-kr"
        ]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise ValueError("읽을 수 없는 TXT 파일입니다.")

    # DOCX
    elif lower_path.endswith(".docx"):

        doc = Document(file_path)

        text = ""

        for p in doc.paragraphs:
            text += p.text + "\n"

        return text

    # PDF
    elif lower_path.endswith(".pdf"):

        doc = fitz.open(file_path)

        text = ""

        for page in doc:
            text += page.get_text()

        return text

    # IMAGE
    elif lower_path.endswith((".png", ".jpg", ".jpeg")):

        image = Image.open(file_path)

        text = pytesseract.image_to_string(
            image,
            lang="kor+eng"
        )

        return text

    raise ValueError(
        "지원하지 않는 파일 형식입니다. (txt, pdf, docx, png, jpg, jpeg)"
    )
