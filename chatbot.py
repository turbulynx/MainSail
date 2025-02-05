from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any
import os
from apiclient import APIClient

# Configuration
CONFIG = {
    'pdf_path': './data/Documentation.pdf',
    'api_key': os.getenv("GOOGLE_API_KEY"),
    'model': {
        'chat': "gemini-pro",
        'embedding': "models/embedding-001"
    },
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'temperature': 0
}


class ContextAwareAgent:
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """
        Initialize the agent with PDF context and API client.

        Args:
            config: Configuration dictionary containing all necessary parameters
        """
        self.config = config
        self.client = APIClient()
        self.docs = []
        self.setup_context()
        self.setup_agent()

    def setup_context(self):
        """Load and process PDF document for context"""
        # Load PDF
        loader = PyPDFLoader(self.config['pdf_path'])
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        self.docs = text_splitter.split_documents(documents)

    def _search_context(self, query: str) -> str:
        """
        Search the PDF context using simple keyword matching
        since the document is small
        """
        relevant_docs = []
        query_terms = query.lower().split()

        for doc in self.docs:
            content = doc.page_content.lower()
            if any(term in content for term in query_terms):
                relevant_docs.append(doc)

        return " ".join([doc.page_content for doc in relevant_docs])

    def setup_agent(self):
        """Set up the React agent with tools"""
        # Initialize language model
        llm = ChatGoogleGenerativeAI(
            temperature=self.config['temperature'],
            model=self.config['model']['chat'],
            google_api_key=self.config['api_key']
        )

        # Create tools
        tools = [
            Tool(
                name="Search Context",
                func=self._search_context,
                description="Search the PDF context to understand if API calls are needed"
            ),
            *self.client.get_tools()
        ]

        # Initialize memory
        memory = ConversationBufferMemory(memory_key="chat_history")

        # Initialize agent
        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory
        )

    def run(self, query: str) -> str:
        """
        Process a user query using the agent.

        Args:
            query: User's question or request

        Returns:
            Agent's response
        """
        return self.agent.run(query)


# Example usage
if __name__ == "__main__":
    # Initialize agent with default configuration
    agent = ContextAwareAgent()

    print("Welcome to the Context-Aware Agent!")
    print("Enter your queries and type 'exit' to quit.")
    print("-" * 50)

    while True:
        query = input("\nEnter your query: ").strip()

        if query.lower() == 'exit':
            print("\nThank you for using the Context-Aware Agent. Goodbye!")
            break

        try:
            response = agent.run(query)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"\nError processing query: {str(e)}")