import io
import os

import fitz
from PIL import Image

from azure_translator import translate_texts

# PyMuPDF에 내장된 CJK 폰트 코드. 별도 폰트 파일 없이 바로 임베딩된다.
_CJK_FONTS = {
    "zh": "china-s",
    "zh-Hans": "china-s",
    "zh-Hant": "china-t",
    "ja": "japan",
    "ko": "korea",
}

# Vietnamese/Kazakh 등 라틴 확장·키릴 문자는 base14(helv)로는 글자가 깨진다.
# 레포에 번들한 NotoSans(라틴+베트남어+키릴 커버)를 비-CJK 언어 기본 폰트로 쓴다.
# .env의 LAYOUT_FONT_FILE을 지정하면 그 TTF가 모든 언어에서 최우선으로 사용된다.
_ENV_FONT_FILE = "LAYOUT_FONT_FILE"
_BUNDLED_FONT = os.path.join(os.path.dirname(__file__), "fonts", "NotoSans-Regular.ttf")


def _srgb_to_rgb(color):
    """PyMuPDF span color(정수 sRGB)를 0~1 RGB 튜플로 변환한다."""
    if not isinstance(color, int):
        return (0.0, 0.0, 0.0)

    red = (color >> 16) & 255
    green = (color >> 8) & 255
    blue = color & 255

    return (red / 255.0, green / 255.0, blue / 255.0)


def _resolve_font(target_code):
    """대상 언어에 맞는 (fontname, fontfile)을 고른다.

    - 사용자가 LAYOUT_FONT_FILE을 주면 그 TTF를 최우선 사용(모든 언어 커버 가능).
    - 중국어/일본어/한국어는 PyMuPDF 내장 CJK 폰트 사용.
    - 그 외(영어 등)는 base14 'helv'.
    """
    font_file = os.getenv(_ENV_FONT_FILE)
    if font_file and os.path.exists(font_file):
        return "F0", font_file

    if target_code in _CJK_FONTS:
        return _CJK_FONTS[target_code], None

    # 영어/베트남어/카자흐어 등은 번들 NotoSans 사용. 없으면 base14 helv로 폴백.
    if os.path.exists(_BUNDLED_FONT):
        return "F0", _BUNDLED_FONT

    return "helv", None


def _build_font(fontname, fontfile):
    """글자 폭 측정을 위한 fitz.Font 객체."""
    if fontfile:
        return fitz.Font(fontfile=fontfile)

    return fitz.Font(fontname=fontname)


def _collect_spans(page):
    """페이지의 텍스트 span(줄 조각) 목록을 좌표/폰트 정보와 함께 모은다."""
    data = page.get_text("dict")
    spans = []

    for block in data.get("blocks", []):
        # type 0만 텍스트 블록. 이미지(type 1)는 건드리지 않아 배경/로고가 보존된다.
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    spans.append(span)

    return spans


def _translate_page(page, target_code, fontname, fontfile, measure_font):
    """페이지 한 장의 텍스트를 같은 위치/색/폰트로 번역문으로 교체한다."""
    spans = _collect_spans(page)
    if not spans:
        return

    translated = translate_texts(
        [span["text"] for span in spans],
        target_code,
    )

    # 1) 원문 글자 제거: 각 span 영역을 흰색으로 가린다.
    for span in spans:
        page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))

    page.apply_redactions()

    # 2) 같은 위치(baseline origin)에 번역문 삽입.
    for span, text in zip(spans, translated):
        if not text.strip():
            continue

        rect = fitz.Rect(span["bbox"])
        size = span.get("size", 11) or 11
        color = _srgb_to_rgb(span.get("color", 0))
        origin = fitz.Point(span["origin"])

        # 번역문이 원본 폭을 넘으면 폰트 크기를 비례 축소(최소 4pt).
        text_width = measure_font.text_length(text, fontsize=size)
        available = rect.width or text_width
        font_size = size

        if text_width > available and text_width > 0:
            font_size = max(4.0, size * (available / text_width))

        page.insert_text(
            origin,
            text,
            fontsize=font_size,
            fontname=fontname,
            fontfile=fontfile,
            color=color,
        )


def translate_pdf_layout(pdf_bytes, target_code):
    """텍스트형 PDF의 레이아웃/색/위치를 유지한 채 글자만 번역한다.

    각 span을 원문 위치에서 지우고(redact), 같은 baseline에 번역문을 그린다.
    번역문이 원문보다 길면 원본 폭에 맞게 폰트 크기를 줄여 한 줄을 유지한다.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    fontname, fontfile = _resolve_font(target_code)
    measure_font = _build_font(fontname, fontfile)

    try:
        for page in doc:
            _translate_page(page, target_code, fontname, fontfile, measure_font)

        return doc.tobytes(garbage=4, deflate=True)

    finally:
        doc.close()


def _fit_canvas_image(pixmap, canvas_w, canvas_h):
    """렌더된 페이지를 target 규격 캔버스에 비율 유지로 넣은 PIL 이미지(여백 흰색)를 돌려준다."""
    source = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    scale = min(canvas_w / source.width, canvas_h / source.height)
    new_w = max(1, int(source.width * scale))
    new_h = max(1, int(source.height * scale))
    resized = source.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(resized, ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2))

    return canvas


def _fit_on_canvas(pixmap, canvas_w, canvas_h):
    """_fit_canvas_image 결과를 PNG 바이트로."""
    buffer = io.BytesIO()
    _fit_canvas_image(pixmap, canvas_w, canvas_h).save(buffer, format="PNG")

    return buffer.getvalue()


def render_translated_page_png(
    pdf_bytes,
    target_code,
    page_index=0,
    dpi=150,
    canvas_w=None,
    canvas_h=None,
):
    """PDF의 특정 페이지(기본 첫 페이지)만 번역해 PNG 이미지 바이트로 렌더한다.

    canvas_w/canvas_h가 주어지면 그 규격 캔버스에 비율 유지로 맞추고(여백은 흰색),
    없으면 원본 페이지 크기 그대로 렌더한다.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    fontname, fontfile = _resolve_font(target_code)
    measure_font = _build_font(fontname, fontfile)

    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError("page_index가 PDF 페이지 범위를 벗어났습니다.")

        page = doc[page_index]
        _translate_page(page, target_code, fontname, fontfile, measure_font)

        pixmap = page.get_pixmap(dpi=dpi)

        if canvas_w and canvas_h:
            return _fit_on_canvas(pixmap, int(canvas_w), int(canvas_h))

        return pixmap.tobytes("png")

    finally:
        doc.close()


def render_translated_pages_combined_png(
    pdf_bytes,
    target_code,
    canvas_w,
    canvas_h,
    dpi=150,
    gap=24,
):
    """PDF 전체 페이지를 번역 후 각 규격 캔버스에 맞추고, 세로로 이어붙인 PNG 한 장으로 반환한다."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    fontname, fontfile = _resolve_font(target_code)
    measure_font = _build_font(fontname, fontfile)

    try:
        canvas_w = int(canvas_w)
        canvas_h = int(canvas_h)
        page_images = []

        for page in doc:
            _translate_page(page, target_code, fontname, fontfile, measure_font)
            pixmap = page.get_pixmap(dpi=dpi)
            page_images.append(_fit_canvas_image(pixmap, canvas_w, canvas_h))

        if not page_images:
            raise ValueError("PDF에 페이지가 없습니다.")

        total_h = canvas_h * len(page_images) + gap * (len(page_images) - 1)
        combined = Image.new("RGB", (canvas_w, total_h), "white")

        offset_y = 0
        for image in page_images:
            combined.paste(image, (0, offset_y))
            offset_y += canvas_h + gap

        buffer = io.BytesIO()
        combined.save(buffer, format="PNG")

        return buffer.getvalue()

    finally:
        doc.close()
