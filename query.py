import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from colorama import Fore, init


@dataclass
class ChatConfig:
    """Configuration for chat application"""
    api_key: str
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "walletmanager"
    model_name: str = "gemini-pro"
    embedding_model: str = "models/embedding-001"


class MessageHandler:
    """Handles message history and formatting"""

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def get_formatted_history(self) -> str:
        if not self.messages:
            return "No previous conversation history."
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in self.messages
        )


class VectorStoreManager:
    """Manages vector store operations"""

    def __init__(self, config: ChatConfig):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.embedding_model)
        self.client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=config.collection_name,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever()

    def get_relevant_documents(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        self._print_debug_info(docs)
        return "\n".join(doc.page_content for doc in docs)

    def _print_debug_info(self, docs: List[Any]) -> None:
        print("\nRetrieved Documents:")
        print("-" * 40)
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 40)


class QueryClassifier:
    """Classifies queries based on their requirements"""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        template = """Analyze if the following question:
                    1. Needs information from a knowledge base to answer accurately
                    2. Refers to or requires chat history context
                    
                    Question: {question}
                    
                    Respond with a JSON object using this exact format:
                    {{"knowledge": boolean, "history": boolean}}"""

        self.chain = (
                ChatPromptTemplate.from_template(template)
                | llm
                | StrOutputParser()
        )

    def classify(self, query: str) -> Dict[str, bool]:
        try:
            result = self.chain.invoke({"question": query})
            return self._parse_classification(result)
        except Exception as e:
            print(f"Classification error: {e}")
            return {"knowledge": False, "history": False}

    def _parse_classification(self, classification_str: str) -> Dict[str, bool]:
        try:
            cleaned = classification_str.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)
            if not isinstance(result, dict) or 'knowledge' not in result or 'history' not in result:
                return {"knowledge": False, "history": False}

            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {"knowledge": False, "history": False}


class ChatBot:
    """Main chatbot class that orchestrates the conversation"""

    def __init__(self, config: ChatConfig):
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, api_key=config.api_key)
        self.vector_store = VectorStoreManager(config)
        self.classifier = QueryClassifier(self.llm)
        self.message_histories: Dict[str, MessageHandler] = {}
        self._setup_chains()

    def _setup_chains(self) -> None:
        self.conversation_chain = (
                ChatPromptTemplate.from_messages([
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ])
                | self.llm
                | StrOutputParser()
        )

        self.runnable_with_history = RunnableWithMessageHistory(
            self.conversation_chain,
            self._get_message_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def _get_message_history(self, session_id: str) -> MessageHandler:
        if session_id not in self.message_histories:
            self.message_histories[session_id] = MessageHandler()
        return self.message_histories[session_id]

    def _create_retrieval_chain(self, history_obj: MessageHandler):
        doc_template = """Here is some relevant information:

{context}

Chat History:
{history}

Based on this, please answer: {question}"""
        doc_prompt = ChatPromptTemplate.from_template(doc_template)

        return (
                {
                    "context": lambda x: self.vector_store.get_relevant_documents(x),
                    "history": lambda x: history_obj.get_formatted_history(),
                    "question": lambda x: x
                }
                | doc_prompt
                | self.llm
                | StrOutputParser()
        )

    def process_query(self, query: str) -> str:
        try:
            classification = self.classifier.classify(query)
            print(f"\nQuery classification: {classification}")

            history = self._get_message_history("history")

            if classification["knowledge"]:
                retrieval_chain = self._create_retrieval_chain(history)
                response = retrieval_chain.invoke(query)
            elif classification["history"]:
                response = self.runnable_with_history.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": "history"}}
                )
            else:
                response = self.llm.invoke(query).content

            history.add_user_message(query)
            history.add_ai_message(response)

            return response
        except Exception as e:
            raise RuntimeError(f"Error processing query: {str(e)}")


def main():
    """Main function to run the chatbot"""
    init()
    load_dotenv()

    config = ChatConfig(api_key=os.getenv("GOOGLE_API_KEY"))
    chatbot = ChatBot(config)

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        try:
            response = chatbot.process_query(query)
            print(f"{Fore.RED}\nResponse:\n{response}{Fore.RESET}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()