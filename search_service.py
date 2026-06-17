import os
from functools import lru_cache

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

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


@lru_cache(maxsize=1)
def get_docs_search_client():
    return SearchClient(
        endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
        index_name=get_required_env("AZURE_SEARCH_INDEX_DOCS"),
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


def search_documents_hybrid(query, query_vector, top=5):
    vector_queries = [
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="title_vector"
        ),
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="content_vector"
        )
    ]

    results = get_docs_search_client().search(
        search_text=query,
        vector_queries=vector_queries,
        select=["id", "title", "category", "content", "source_file"],
        top=top
    )

    documents = []

    for item in results:
        documents.append({
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "content": item.get("content", ""),
            "source_file": item.get("source_file", ""),
            "score": item.get("@search.score")
        })

    return documents
