from abc import ABC, abstractmethod
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingsProvider(ABC):
    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def get_vector_store(self, documents):
        pass

    @abstractmethod
    def get_chat_model(self):
        pass

class GoogleProvider(EmbeddingsProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def get_vector_store(self, documents):
        client = chromadb.HttpClient(host="localhost", port=8000)
        return Chroma(
            client=client,
            collection_name="hare",
            embedding_function=self.get_embeddings()
        )

    def get_chat_model(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=self.model_name, api_key=self.api_key)

class OllamaProvider(EmbeddingsProvider):
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.model_name)

    def get_vector_store(self, documents):
        client = chromadb.HttpClient(host="localhost", port=8000)
        return Chroma(
            client=client,
            collection_name="tcp",
            embedding_function=self.get_embeddings()
        )

    def get_chat_model(self):
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=self.model_name)