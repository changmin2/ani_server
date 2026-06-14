import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(
        os.getenv("AZURE_SEARCH_KEY")
    )
)


def search_finance_terms(text):

    results = client.search(
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