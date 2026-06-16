from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

import os

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


documents = [
    {
        "id": "1",
        "term_ko": "기준 금리",
        "term_en": "based rate",
        "description": "Banking term"
    },
    {
        "id": "2",
        "term_ko": "예금",
        "term_en": "deposit",
        "description": "Banking term"
    },
    {
        "id": "3",
        "term_ko": "적금",
        "term_en": "installment savings",
        "description": "Banking term"
    }
]


def upload_documents():
    client = SearchClient(
        endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
        index_name=get_required_env("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(
            get_required_env("AZURE_SEARCH_KEY")
        )
    )

    return client.upload_documents(documents)


if __name__ == "__main__":
    result = upload_documents()
    print(result)
