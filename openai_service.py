import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


@lru_cache(maxsize=1)
def get_openai_client():
    return OpenAI(
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),
        base_url=get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    )


def improve_translation(original_text, machine_translation, glossary_terms, target_lang):

    glossary_text = ""
    for t in glossary_terms:
        glossary_text += f"{t['ko']} = {t['en']}\n"

    prompt = f"""
You are a banking translation expert.

Target language: {target_lang}

Glossary:
{glossary_text}

Original:
{original_text}

Machine translation:
{machine_translation}

Return only final translation.
"""
    response = get_openai_client().chat.completions.create(
        model=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
