import os
from typing import Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
from langchain_chroma import Chroma

class DocumentSearchAgent:
    """
    A RAG-based (Retrieval-Augmented Generation) Agent that can answer questions
    by searching through a local document store.
    Designed to be integrated into a FastAPI service.
    """

    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initializes the agent with necessary LLM configuration and vector store.

        :param openai_api_key: The API key for OpenAI services.
        :param model_name: The name of the OpenAI model to use.
        """
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided.")

        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize the Language Model (temperature=0 for stable, deterministic testing)
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # Initialize Embeddings and an in-memory Vector Store (Chroma)
        self.embeddings = OpenAIEmbeddings()
        self._last_retrieved_documents: list[Document] = []
        self.vector_store = Chroma(
            collection_name="local_documents",
            embedding_function=self.embeddings
        )

        # Set up the execution engine
        self.agent_executor = self._create_agent_executor()

    def upload_document(self, text: str, source: str) -> None:
        document = Document(
            page_content=text,
            metadata={"source": source},
        )
        self.vector_store.add_documents([document])

    def retrieve_documents(self, query: str, k: int = 2) -> list[Document]:
        """Retrieve documents using the same vector-store settings as the agent."""
        return self.vector_store.similarity_search(query, k=k)

    def ask_with_context(self, user_input: str, chat_history: list[Any] = None) -> dict[str, Any]:
        """Return the generated answer together with contexts for evaluation."""
        self._last_retrieved_documents = []
        result = self.ask(user_input=user_input, chat_history=chat_history)
        documents = self._last_retrieved_documents
        return {
            "output": result.get("output", ""),
            "contexts": [document.page_content for document in documents],
            "sources": [document.metadata.get("source", "") for document in documents],
        }

    def _create_agent_executor(self) -> AgentExecutor:
        """
        Internal method to configure tools, prompts, and build the Agent Executor.

        :return: An instance of AgentExecutor.
        """

        @tool
        def search_documents(query: str) -> str:
            """
            Search for relevant information within the uploaded documents store.
            Use this tool when the user asks specific questions about company rules,
            internal guides, or uploaded data.
            """
            # Retrieve top 2 most relevant document snippets
            docs = self.vector_store.similarity_search(query, k=2)
            self._last_retrieved_documents = docs
            if not docs:
                return "No relevant documents found."

            # Combine snippets into a single context string
            results = []

            for doc in docs:
                results.append(
                    f"Source: {doc.metadata['source']}\n"
                    f"Content: {doc.page_content}"
                )

            return "\n---\n".join(results)

        # Define the toolset available to the agent
        tools = [search_documents]

        # Define the system prompt guiding the agent's behavior
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful assistant. You must answer user questions based ONLY "
                "on the information retrieved via the available tools. If the tools do not "
                "provide enough information, politely inform the user that you do not know the answer. "
                "Always be factual and do not hallucinate."
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Construct the OpenAI Tools Agent
        agent = create_openai_tools_agent(self.llm, tools, prompt)

        # Return the executor responsible for running the agent loop
        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    def ask(self, user_input: str, chat_history: list[Any] = None) -> dict[str, Any]:
        """
        Sends a user query to the agent and returns the generated answer.

        :param user_input: The question or statement from the user.
        :param chat_history: Optional list of past messages for context retention.
        :return: A dictionary containing the 'output' and execution details.
        """
        history = chat_history or []
        return self.agent_executor.invoke({"input": user_input, "chat_history": history})
