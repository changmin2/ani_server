import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def get_vision_client():
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")

    if not endpoint or not key:
        raise Exception("AZURE_VISION_ENDPOINT 또는 AZURE_VISION_KEY가 .env에 없습니다.")

    return ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )


def polygon_to_box(polygon):
    xs = [point.x for point in polygon]
    ys = [point.y for point in polygon]

    return {
        "x": min(xs),
        "y": min(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "polygon": [
            {
                "x": point.x,
                "y": point.y
            }
            for point in polygon
        ]
    }


def extract_ocr_texts(image_bytes: bytes):
    client = get_vision_client()

    result = client.analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.READ]
    )

    texts = []

    if result.read is None:
        return texts

    for block in result.read.blocks:
        for line in block.lines:
            box = polygon_to_box(line.bounding_polygon)

            texts.append({
                "text": line.text,
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "polygon": box["polygon"]
            })

    return texts