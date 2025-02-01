import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
model_name: str = "gemini-pro"
genai.configure(api_key=google_api_key)

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    print(file_path)
    return loader.load()

def load_text(file_path):
    loader = TextLoader(file_path)
    return loader.load()

def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print(f"loading... {file_path}")
        if filename.endswith(".pdf"):
            documents.extend(load_pdf(file_path))
        elif filename.endswith(".txt"):
            documents.extend(load_text(file_path))
    return documents

def create_and_save_vectorstore(documents, embeddings):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"Total documents to index: {len(docs)}")
    client = chromadb.HttpClient(host="localhost", port=8000)
    db = Chroma.from_documents(docs, embeddings,  client=client, collection_name="hare")
    return db

data_dir = "./data"
all_documents = load_documents_from_directory(data_dir)

if all_documents:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = create_and_save_vectorstore(all_documents, embeddings)

    query = "What is document about?"
    similar_docs = db.similarity_search(query)
    for doc in similar_docs:
        print(doc.page_content)
        print("-" * 20)