# DeepEval Training Sandbox

Educational project for learning and practicing **DeepEval** with a real Python RAG application.

The project contains a small FastAPI service with a Retrieval-Augmented Generation (RAG) agent. The agent loads local knowledge-base documents into a Chroma vector store and uses LangChain and OpenAI models to answer questions based on the retrieved documents.

The repository also contains examples of unit and integration testing. DeepEval evaluation tests will be added as part of the training.

## Requirements

- Python 3.10+
- PostgreSQL (required for integration tests)
- An OpenAI API key
- Internet access for OpenAI model and embedding requests

The project was prepared primarily for Windows and PyCharm, but the Python commands can also be run from other IDEs or terminals.

## Technology stack

- FastAPI
- Uvicorn
- LangChain
- OpenAI
- Chroma
- PostgreSQL
- SQLAlchemy
- Pytest
- pytest-asyncio
- DeepEval (for evaluation exercises)

## Project structure

```text
DeepEvalSandbox/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   ├── it_practice.py
│   ├── ut_practice.py
│   │
│   ├── knowledge_base/
│   │   ├── engineering_stack.txt
│   │   ├── hr_remote_work.txt
│   │   ├── it_laptop_request.txt
│   │   └── office_kitchen.txt
│   │
│   └── tests/
│       ├── __init__.py
│       │
│       ├── unit/
│       └── integration/
│
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── __init__.py
```

## Setup

Run all commands from the project root.

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure the OpenAI API key

Create a `.env` file and add:

```text
OPENAI_API_KEY=your-openai-api-key
```

The API key is required to run the RAG agent.

### 4. PostgreSQL

PostgreSQL is required for the integration tests.

Make sure PostgreSQL is installed and running locally, and that the database configuration used by the project is available.

## Run the application

Start the FastAPI application:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

The application loads the documents from `app/knowledge_base/` when it starts.

## Run tests

### Unit tests

```bash
pytest app/tests/unit
```

### Integration tests

Make sure PostgreSQL is running, then:

```bash
pytest app/tests/integration
```

## DeepEval

This repository is used as a training sandbox for learning how to evaluate a real RAG application with **DeepEval**.

DeepEval evaluation code and datasets will be added incrementally as part of the training exercises.