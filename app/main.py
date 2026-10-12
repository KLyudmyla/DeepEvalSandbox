import os
from fastapi import Depends, FastAPI, HTTPException
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv()

from app.agent import DocumentSearchAgent

app = FastAPI(
    title="LLM RAG Agent Service",
    description="A simple FastAPI service wrapper around our Document Search Agent for testing purposes."
)

# Initialize the agent. It will look for OPENAI_API_KEY in environment variables.
# We do this at the module level so the agent persists across API calls.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set!")

agent = DocumentSearchAgent(openai_api_key=api_key)

# Mock corporate knowledge base to preload into our in-memory vector store on startup
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"


@app.on_event("startup")
def preload_documents():
    """
    Loads knowledge base documents into the vector store when the server starts.
    """
    print("Loading knowledge base documents...")
    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        agent.upload_document(
            text=text,
            source=file_path.name,
        )

    print("Documents successfully loaded!")


# Define the request structure using Pydantic for validation
class QueryRequest(BaseModel):
    question: str


# Define the response structure
class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    """
    Endpoint to send questions to the LLM agent.
    The agent will search through documents and formulate an answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Call our agent's ask method
        agent_response = agent.ask(user_input=request.question)
        return QueryResponse(answer=agent_response["output"])
    except Exception as e:
        # Handle unexpected errors gracefully
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
