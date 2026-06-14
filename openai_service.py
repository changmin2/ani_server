import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
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
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content