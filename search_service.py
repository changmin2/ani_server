import os
from functools import lru_cache

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


@lru_cache(maxsize=1)
def get_search_client():
    return SearchClient(
        endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
        index_name=get_required_env("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(
            get_required_env("AZURE_SEARCH_KEY")
        )
    )


def search_finance_terms(text):

    results = get_search_client().search(
        search_text=text,
        top=20
    )

    glossary = []

    for item in results:

        glossary.append({
            "ko": item["term_ko"],
            "en": item["term_en"],
            "description": item.get(
                "description",
                ""
            )
        })

    return glossary
