import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")

client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

results = client.search(
    search_text="기준금리",
    top=5
)

for item in results:
    print(item)