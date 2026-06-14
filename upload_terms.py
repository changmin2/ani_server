from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

endpoint = "https://anisearch.search.windows.net"
index_name = "finance"
key = "wcGA1IyGejGBMN7oAVD7OYekmBikhu6TjZGZz4eQljAzSeBbelRU"

client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

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

result = client.upload_documents(documents)

print(result)