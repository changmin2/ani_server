import os
import re
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
    # 기존 금융 용어집 인덱스용 클라이언트.
    # 현재 문서 분석 화면에서는 docs 인덱스를 주로 사용하지만, 기존 코드 호환을 위해 유지한다.
    return SearchClient(
        endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
        index_name=get_required_env("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(
            get_required_env("AZURE_SEARCH_KEY")
        )
    )


@lru_cache(maxsize=1)
def get_docs_search_client():
    # sample_docs.json을 업로드한 문서 분석용 인덱스 클라이언트.
    # title/content 텍스트 필드와 title_vector/content_vector 벡터 필드를 함께 사용한다.
    return SearchClient(
        endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
        index_name=get_required_env("AZURE_SEARCH_INDEX_DOCS"),
        credential=AzureKeyCredential(
            get_required_env("AZURE_SEARCH_KEY")
        )
    )


def search_finance_terms(text):
    # 기존 번역 기능에서 사용하던 금융 용어 검색 함수.
    # 현재 main.py에서는 제거되었지만 다른 테스트 코드에서 쓸 수 있어 그대로 둔다.

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


def search_finance_terms_hybrid(query, query_vector, top=12):
    vector_queries = [
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top,
            fields="search_vector"
        )
    ]

    results = get_search_client().search(
        search_text=query,
        vector_queries=vector_queries,
        select=[
            "id",
            "term_ko",
            "term_en",
            "term_vi",
            "term_zh",
            "term_kk",
            "description",
            "search_text"
        ],
        top=max(top * 5, top)
    )

    terms = []
    seen_terms = set()

    for item in results:
        normalized_ko = normalize_term_text(item.get("term_ko", ""))

        if normalized_ko in seen_terms:
            continue

        seen_terms.add(normalized_ko)
        terms.append(to_finance_term(item, match_type="hybrid"))

        if len(terms) >= top:
            break

    return terms


def normalize_term_text(text):
    return re.sub(r"\s+", "", str(text).casefold())


def to_finance_term(item, match_type=None):
    term = {
        "id": item.get("id", ""),
        "term_ko": item.get("term_ko", ""),
        "term_en": item.get("term_en", ""),
        "term_vi": item.get("term_vi", ""),
        "term_zh": item.get("term_zh", ""),
        "term_kk": item.get("term_kk", ""),
        "description": item.get("description", ""),
        "search_text": item.get("search_text", ""),
        "score": item.get("@search.score")
    }

    if match_type:
        term["match_type"] = match_type

    return term


def find_finance_terms_in_text(query, top=12):
    query_text = str(query)
    normalized_query = normalize_term_text(query_text)

    if not normalized_query:
        return []

    results = get_search_client().search(
        search_text="*",
        select=[
            "id",
            "term_ko",
            "term_en",
            "term_vi",
            "term_zh",
            "term_kk",
            "description",
            "search_text"
        ],
        top=1000
    )

    matched_terms = []

    for item in results:
        matched_values = [
            item.get("term_ko", ""),
            item.get("term_en", ""),
            item.get("term_vi", ""),
            item.get("term_zh", ""),
            item.get("term_kk", "")
        ]

        normalized_values = [
            normalize_term_text(value)
            for value in matched_values
            if normalize_term_text(value)
        ]

        positions = [
            normalized_query.find(value)
            for value in normalized_values
            if value in normalized_query
        ]

        if not positions:
            continue

        longest_match_length = max(len(value) for value in normalized_values if value in normalized_query)
        term = to_finance_term(item, match_type="exact")
        term["_match_position"] = min(positions)
        term["_match_length"] = longest_match_length
        matched_terms.append(term)

    matched_terms.sort(key=lambda term: (term["_match_position"], -term["_match_length"]))

    deduped_terms = []
    covered_terms = set()

    for term in matched_terms:
        normalized_ko = normalize_term_text(term.get("term_ko", ""))

        if normalized_ko and any(normalized_ko in covered for covered in covered_terms):
            continue

        covered_terms.add(normalized_ko)
        term.pop("_match_position", None)
        term.pop("_match_length", None)
        deduped_terms.append(term)

        if len(deduped_terms) >= top:
            break

    return deduped_terms


def search_documents_hybrid(query, query_vector, top=5):
    # Azure AI Search 하이브리드 검색:
    # - search_text=query: 키워드 기반 검색
    # - vector_queries: 입력 원문 임베딩과 문서 벡터 필드 간 유사도 검색
    # 두 검색 신호를 함께 사용해 짧은 문장/긴 문서 모두에서 관련 문서를 찾기 쉽게 한다.
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
        # 프론트의 "AI 추천 근거 보기" 패널에서 바로 렌더링하기 쉬운 형태로 정리한다.
        documents.append({
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "content": item.get("content", ""),
            "source_file": item.get("source_file", ""),
            "score": item.get("@search.score")
        })

    return documents
