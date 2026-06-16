import os
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def translate(text, target_lang):
    url = f"{get_required_env('AZURE_TRANSLATOR_ENDPOINT')}/translate"

    params = {
        "api-version": "3.0",
        "to": target_lang
    }

    headers = {
        "Ocp-Apim-Subscription-Key": get_required_env("AZURE_TRANSLATOR_KEY"),
        "Ocp-Apim-Subscription-Region": get_required_env("AZURE_TRANSLATOR_REGION"),
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }

    body = [{"text": text}]

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=body,
        timeout=10
    )

    response.raise_for_status()

    return response.json()[0]["translations"][0]["text"]
