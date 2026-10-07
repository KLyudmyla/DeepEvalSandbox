import os
from fastapi import Depends, FastAPI, HTTPException
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv()

from app.agent import DocumentSearchAgent
from app.it_practice import engine, Base, get_db, User, Article


Base.metadata.create_all(bind=engine)

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


@app.post("/users/", status_code=201)
def create_user(username: str, email: str, db: Session = Depends(get_db)):
    """
    Creates a new user in the database.

    TESTING HINTS:
    - Positive: Verify a new user is successfully saved and returned with status 201.
    - Negative: Verify that registering an existing email raises HTTP 400 ("Email already registered").
    """
    db_user = db.query(User).filter(User.email == email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(username=username, email=email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"id": new_user.id, "username": new_user.username, "email": new_user.email}


@app.post("/articles/", status_code=201)
def create_article(title: str, content: str, owner_id: int, db: Session = Depends(get_db)):
    """
    Creates a new article linked to an existing user (author).

    TESTING HINTS:
    - Positive: Verify article creation when a valid owner_id is provided.
    - Negative: Verify that providing a non-existent owner_id raises HTTP 404 ("User not found").
    """
    author = db.query(User).filter(User.id == owner_id).first()
    if not author:
        raise HTTPException(status_code=404, detail="User not found")

    new_article = Article(title=title, content=content, owner_id=owner_id)
    db.add(new_article)
    db.commit()
    db.refresh(new_article)
    return {"id": new_article.id, "title": new_article.title, "owner_id": new_article.owner_id}


@app.get("/users/", status_code=200)
def get_users(db: Session = Depends(get_db)):
    """
    Retrieves all users from the database.

    TESTING HINTS:
    - Test empty state: Verify it returns an empty list [] when no users exist.
    - Test populated state: Verify it returns all created users with correct fields.
    """
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username, "email": u.email} for u in users]


@app.get("/articles/", status_code=200)
def get_articles(db: Session = Depends(get_db)):
    """
    Retrieves all articles from the database.

    TESTING HINTS:
    - Test empty state: Verify it returns an empty list [].
    - Test populated state: Verify it returns all articles with their associated owner_ids.
    """
    articles = db.query(Article).all()
    return [{"id": a.id, "title": a.title, "content": a.content, "owner_id": a.owner_id} for a in articles]
