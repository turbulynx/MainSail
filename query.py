import os
import json
from typing import List
from langchain_core.messages import BaseMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from colorama import Fore, init

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

client = chromadb.HttpClient(host="localhost", port=8000)
vectorstore = Chroma(client=client, collection_name="tcp", embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(model='gemini-pro', api_key=google_api_key)

retriever = vectorstore.as_retriever()

classifier_template = """Analyze if the following question:
1. Needs information from a knowledge base to answer accurately
2. Refers to or requires chat history context

Question: {question}

Respond with a JSON object using this exact format:
{{"knowledge": boolean, "history": boolean}}

For example:
{{"knowledge": true, "history": false}}"""

classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
classifier_chain = classifier_prompt | llm | StrOutputParser()

doc_template = """Here is some relevant information:

{context}

Chat History:
{history}

Based on this, please answer: {question}"""

doc_prompt = ChatPromptTemplate.from_template(doc_template)


def get_context_with_debug(query):
    docs = retriever.get_relevant_documents(query)
    print("\nRetrieved Documents:")
    print("-" * 40)
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)
    return "\n".join(doc.page_content for doc in docs)


def format_history(messages):
    if not messages:
        return "No previous conversation history."
    formatted = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


class MessageHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add multiple messages to the history."""
        self.messages.extend(messages)

    def get_formatted_history(self):
        return format_history(self.messages)


message_histories = {}


def get_message_history(session_id: str):
    if session_id not in message_histories:
        message_histories[session_id] = MessageHistory()
    return message_histories[session_id]


conversation_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

conversation_chain = conversation_prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    conversation_chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history"
)


def create_retrieval_chain(history_obj):
    return (
            {
                "context": lambda x: get_context_with_debug(x),
                "history": lambda x: history_obj.get_formatted_history(),
                "question": lambda x: x
            }
            | doc_prompt
            | llm
            | StrOutputParser()
    )


def parse_classification(classification_str):
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


while True:
    query = input("Enter your query (type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Exiting...")
        break

    try:
        classification_str = classifier_chain.invoke({"question": query})
        print(f"\nQuery classification: {classification_str}")

        needs = parse_classification(classification_str)
        history = get_message_history("history")

        if needs["knowledge"]:
            retrieval_chain = create_retrieval_chain(history)
            response = retrieval_chain.invoke(query)
        elif needs["history"]:
            response = runnable_with_history.invoke(
                {"input": query},
                config={"configurable": {"session_id": "history"}}
            )
        else:
            response = llm.invoke(query).content

        # Update history
        history.add_user_message(query)
        history.add_ai_message(response)

        print(f"{Fore.RED}\nResponse:\n{response}{Fore.RESET}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again.")