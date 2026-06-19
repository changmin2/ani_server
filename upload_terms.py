import json
import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TERMS_FILE = BASE_DIR / "bank_terms_vi.json"
SEARCH_TEXT_FIELDS = ("term_ko", "term_en", "term_vi", "term_zh", "term_kk")
BATCH_SIZE = 100


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def load_documents():
    with TERMS_FILE.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    if not isinstance(documents, list):
        raise RuntimeError(f"{TERMS_FILE.name} must contain a JSON array")

    return documents


def dedupe_documents(documents):
    deduped_documents = []
    seen_terms = set()

    for document in documents:
        term_key = (
            document.get("term_ko", "").strip(),
            document.get("term_en", "").strip(),
            document.get("term_vi", "").strip(),
            document.get("term_zh", "").strip(),
            document.get("term_kk", "").strip()
        )

        if term_key in seen_terms:
            continue

        seen_terms.add(term_key)
        deduped_documents.append(document)

    print(f"중복 제거: {len(documents)}건 → {len(deduped_documents)}건")

    return deduped_documents


openai_client = OpenAI(
    base_url=get_required_env("AZURE_OPENAI_DOCS_ENDPOINT"),
    api_key=get_required_env("AZURE_OPENAI_DOCS_API_KEY")
)

EMBEDDING_MODEL = get_required_env("AZURE_OPENAI_DOCS_DEPLOYMENT")


def get_embedding(text):
    if not text:
        text = " "

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def build_search_text(document):
    return " ".join(
        str(document.get(field, "")).strip()
        for field in SEARCH_TEXT_FIELDS
        if str(document.get(field, "")).strip()
    )


def prepare_documents(documents):
    for index, document in enumerate(documents, start=1):
        search_text = build_search_text(document)
        document["search_text"] = search_text

        print(f"[{index}/{len(documents)}] 벡터 생성 중: {document.get('id')}")
        document["search_vector"] = get_embedding(search_text)

    return documents


def delete_documents_batch(client, documents, label):
    if not documents:
        return 0

    results = client.delete_documents(documents=documents)
    success_count = sum(1 for item in results if item.succeeded)

    print(f"{label}: {success_count}/{len(documents)}건")

    return success_count


def delete_existing_documents(client):
    # 새 bank_terms_vi.json을 단일 원천으로 사용하기 위해 기존 검색 인덱스 문서를 전부 삭제한다.
    # search_text="*"로 id만 읽어온 뒤 배치 삭제하므로 JSON에 없는 과거 용어도 남지 않는다.
    results = client.search(
        search_text="*",
        select=["id"],
        top=1000
    )
    documents_to_delete = [
        {"id": item["id"]}
        for item in results
        if item.get("id")
    ]

    if not documents_to_delete:
        print("기존 인덱스 문서 없음")
        return 0

    print(f"기존 인덱스 문서 삭제 대상: {len(documents_to_delete)}건")

    deleted_count = 0

    for start in range(0, len(documents_to_delete), BATCH_SIZE):
        batch = documents_to_delete[start:start + BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1
        deleted_count += delete_documents_batch(
            client,
            batch,
            f"기존 인덱스 문서 삭제 배치 {batch_number}"
        )

    return deleted_count


def upload_documents():
    client = SearchClient(
        endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
        index_name=get_required_env("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(
            get_required_env("AZURE_SEARCH_KEY")
        )
    )

    delete_existing_documents(client)

    original_documents = load_documents()
    documents = dedupe_documents(original_documents)
    documents = prepare_documents(documents)
    print(f"업로드 대상 용어 수: {len(documents)}")

    return client.upload_documents(documents=documents)


if __name__ == "__main__":
    result = upload_documents()
    success_count = sum(1 for item in result if item.succeeded)

    print("=" * 50)
    print(f"업로드 성공: {success_count}건")
    print(f"업로드 실패: {len(result) - success_count}건")
    print("=" * 50)
