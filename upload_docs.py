import json
import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from openai import OpenAI

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value



openai_client = OpenAI(
    base_url=get_required_env("AZURE_OPENAI_DOCS_ENDPOINT"),
    api_key=get_required_env("AZURE_OPENAI_DOCS_API_KEY")
)

EMBEDDING_MODEL = get_required_env("AZURE_OPENAI_DOCS_DEPLOYMENT")


def get_embedding(text: str):
    if not text:
        text = " "

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


# Azure AI Search 클라이언트
search_client = SearchClient(
    endpoint=get_required_env("AZURE_SEARCH_ENDPOINT"),
    index_name=get_required_env("AZURE_SEARCH_INDEX_DOCS"),
    credential=AzureKeyCredential(
        get_required_env("AZURE_SEARCH_KEY")
    )
)


with open("sample_docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"업로드 대상 문서 수: {len(docs)}")


for i, doc in enumerate(docs, start=1):
    title = doc.get("title", "")
    content = doc.get("content", "")

    print(f"[{i}/{len(docs)}] 벡터 생성 중: {doc.get('id')}")

    doc["title_vector"] = get_embedding(title)
    doc["content_vector"] = get_embedding(content)


results = search_client.upload_documents(documents=docs)

success_count = sum(1 for r in results if r.succeeded)

print("=" * 50)
print(f"업로드 성공: {success_count}건")
print(f"업로드 실패: {len(docs) - success_count}건")
print("=" * 50)