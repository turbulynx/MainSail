import chromadb
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

client = chromadb.HttpClient(host="localhost", port=8000)
ollama_base_url = "http://localhost:11434"
embeddings = OllamaEmbeddings(base_url=ollama_base_url, model="nomic-embed-text")

db = Chroma(client=client, collection_name="tcp", embedding_function=embeddings)

query = "What is the document about?"
similar_docs = db.similarity_search(query)

print("\nTop matching documents:\n")
for doc in similar_docs:
    print(doc.page_content)
    print("-" * 20)
