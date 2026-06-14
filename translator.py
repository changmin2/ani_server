import os
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("AZURE_TRANSLATOR_KEY")
ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
REGION = os.getenv("AZURE_TRANSLATOR_REGION")


def translate(text, target_lang):
    url = f"{ENDPOINT}/translate"

    params = {
        "api-version": "3.0",
        "to": target_lang
    }

    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Ocp-Apim-Subscription-Region": REGION,
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }

    body = [{"text": text}]

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=body
    )

    response.raise_for_status()

    return response.json()[0]["translations"][0]["text"]