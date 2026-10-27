# DeepEval Training Sandbox

Educational project for learning and practicing **DeepEval** with a real Python RAG application.

The project contains a small FastAPI service with a Retrieval-Augmented Generation (RAG) agent. The agent loads local 
knowledge-base documents into a Chroma vector store and uses LangChain and OpenAI models to answer questions based on
the retrieved documents.

The repository also contains examples of unit testing, a Layer 1 evaluation suite for the RAG ingestion pipeline, and an
implemented Layer 2 evaluation for retrieval quality. DeepEval is included for the next evaluation layers: answer quality
and agent/tool behavior.

## Requirements

- Python 3.12.10
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
│   ├── agent.py                      # RAG agent and Chroma-backed document search
│   ├── main.py                       # FastAPI app and knowledge-base startup loading
│   ├── it_practice.py
│   ├── ut_practice.py
│   ├── knowledge_base/               # Source documents loaded into the vector store
│   │   └── *.txt
│   └── tests/unit/                   # Application unit tests
├── evals/
│   ├── adapters/layer1_adapter.py   # Layer 1 agent and Chroma inspection helpers
│   ├── datasets/golden_dataset_l2.py # Layer 2 retrieval scenarios and expected sources
│   ├── fixtures/layer1/              # Shared isolated-agent fixtures
│   ├── layer1/                       # Knowledge-base loading and integrity tests
│   └── layer2/
│       ├── __main__.py               # CLI entry point for the live evaluation harness
│       ├── config.py                 # Layer 2 models and thresholds
│       ├── deepeval_bridge.py        # DeepEval test-case/metric integration
│       ├── metrics.py                # Deterministic retrieval metrics
│       ├── reporting.py              # Unified reports, gates, and cost telemetry
│       ├── run_layer2.py             # Evaluation orchestration
│       ├── utils.py                  # Dataset validation and retrieval helpers
│       └── tests/                    # Offline regressions plus opt-in live tests
├── .env                              # Local OPENAI_API_KEY configuration
├── .gitignore
├── requirements.txt
├── README.md
└── __init__.py
```

The `.venv/`, `.deepeval/`, `.pytest_cache/`, `__pycache__/`, and `.idea/` directories are local or generated files and 
are intentionally excluded from the source tree above.

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

The integration-test directory is reserved for the PostgreSQL exercises; the current project tree contains the SQLAlchemy
practice module in `app/it_practice.py` but no integration test modules yet.

### Layer 1 evaluation tests

Layer1 tests validate the RAG application's document-ingestion and knowledge-base loading behavior through the real 
`DocumentSearchAgent`. They are intended to catch regressions in the foundation of the RAG pipeline before evaluating 
answer quality. The checks cover:

- loading valid `.txt` documents into the vector store;
- preserving document content exactly, including UTF-8 text;
- preserving each document's source metadata;
- duplicate uploads and updates for an existing source;
- rejecting invalid UTF-8, blank documents, and blank sources without mutating the store;
- loading the application knowledge base at startup and making preloaded documents retrievable; and
- isolating independent evaluation runs from one another.

The tests create a fresh in-memory agent for each scenario, so PostgreSQL is not required.

Run the complete Layer1 suite from the project root:

```bash
python -m pytest evals/layer1 -v
```

Run one Layer1 test module, for example:

```bash
python -m pytest evals/layer1/test_document_loading.py -v
```

Use `-k` to select tests by name, for example:

```bash
python -m pytest evals/layer1 -k duplicate -v
```

Some Layer1 tests intentionally describe requirements that are not implemented yet, such as content-based deduplication 
and blank-document rejection. A failure in one of those cases identifies an application behavior that still needs to be 
implemented; it does not indicate that the test should be removed.

### Layer 2 evaluation tests

Layer 2 evaluates retrieval correctness: whether the real RAG agent returns the relevant knowledge-base documents for a
question, ranks useful documents above distractors, preserves source metadata, and returns no unsupported evidence for
unanswerable questions. The evaluation uses the golden dataset in `evals/datasets/golden_dataset_l2.py`, deterministic
retrieval metrics, and an optional DeepEval GEval judge for semantic relevance. Each run writes a unified JSON report.

The CLI uses the real agent and therefore still requires `OPENAI_API_KEY` even when the judge is disabled; the no-judge mode
only removes the additional semantic-judge calls.

Run the deterministic Layer 2 evaluation without the GEval judge. Report will be created by default:

```bash
python -m evals.layer2 --without-judge
```

Run the full Layer 2 evaluation with the default `gpt-4o-mini` GEval judge. To change the LLL model-> change config.py
(evals/layer2/config.py:41) and __main__.py (evals/layer2/__main__.py:260):

```bash
python -m evals.layer2
```

Useful CLI options:

```bash
python -m evals.layer2 --with-judge --judge-model gpt-4o-mini
python -m evals.layer2 --without-judge --report reports/layer2-deterministic.json
python -m evals.layer2 --max-judge-cost-usd 1.00
```

`--with-judge` explicitly enables the default semantic judge; `--judge-model` selects another judge model; `--report` sets
the JSON output path; and `--max-judge-cost-usd` fails the run if the known judge cost exceeds the specified budget.

Run the offline Layer 2 pytest suite. Live tests are deselected unless explicitly enabled, so this command does not construct
the real agent or call paid external APIs:

```bash
python -m pytest evals/layer2/tests -v
```

Opt in to the live Layer 2 pytest evaluation with the default judge:

```bash
python -m pytest evals/layer2/tests --run-layer2-live -v
```

Run the live pytest evaluation with deterministic metrics only, or apply a judge model and cost limit:

```bash
python -m pytest evals/layer2/tests --run-layer2-live --layer2-without-judge -v
python -m pytest evals/layer2/tests --run-layer2-live --layer2-judge-model gpt-4o-mini --layer2-max-judge-cost-usd 1.00 -v
```

`--run-layer2-live` enables the real-agent tests; `--layer2-without-judge` disables GEval; `--layer2-judge-model` chooses
the judge; and `--layer2-max-judge-cost-usd` applies a safe budget gate. The live pytest fixture runs the full golden dataset
once and exposes each case as an individual pytest result.

## DeepEval

This repository is used as a training sandbox for evaluating a real RAG application with **DeepEval**. Evaluation is 
organized as a pipeline: validate the knowledge base first, then retrieval, then generated answers, and finally the 
agent's tool use and decisions. A failure in an earlier layer should be fixed or triaged before using later-layer scores
to judge the system.

### Evaluation strategy

#### Layer 1 — Knowledge base loading and integrity

This layer verifies that the evidence available to the RAG system is correct before measuring retrieval or answer 
quality. The existing `evals/layer1/` suite exercises the real `DocumentSearchAgent` and its in-memory Chroma store. It 
checks valid `.txt` ingestion, exact content preservation, UTF-8 handling, source metadata, duplicate/update behavior, 
blank-input rejection, startup loading from `app/knowledge_base/`, preloaded-document retrievability, and isolation 
between evaluation runs. These are primarily deterministic contract checks and act as a gate for the LLM-as-a-judge layers.

#### Layer 2 — Relevant document retrieval

This layer measures whether a user question produces the right supporting documents, with useful documents ranked above 
irrelevant ones. Each case should capture the user `input`, the agent's `retrieval_context`, returned source metadata, 
and—when available—an expected answer or set of supporting sources. Use DeepEval's [Contextual Relevancy]
(https://deepeval.com/docs/metrics-contextual-relevancy) for signal-to-noise, 
[Contextual Precision](https://deepeval.com/docs/metrics-contextual-precision) for ranking, and 
[Contextual Recall](https://deepeval.com/docs/metrics-contextual-recall) for coverage against the expected answer. 
Keep exact source-hit assertions alongside these judge-based metrics for stable regression detection.

#### Layer 3 — Answer accuracy and grounding

This layer evaluates the answer produced from the retrieved context. Each case should provide `input`, `actual_output`, 
`retrieval_context`, and `expected_output` for labeled questions. Combine [Answer Relevancy]
(https://deepeval.com/docs/metrics-answer-relevancy) to check that the response addresses the question, [Faithfulness]
(https://deepeval.com/docs/metrics-faithfulness) to check that claims are supported by retrieved documents, and a 
reference-based correctness rubric (for example, GEval) for the expected answer. Include questions whose answer is not 
present in the knowledge base to verify that the agent abstains instead of using unsupported outside knowledge.

#### Layer 4 — Tool selection, execution, and decisions

This layer evaluates the agent loop rather than only its final text. Capture the selected tool, tool arguments, 
execution result, number and order of calls, and the final decision. Use DeepEval's [Tool Correctness]
(https://deepeval.com/docs/metrics-tool-correctness) for expected tool selection and arguments, and add deterministic 
assertions or a custom GEval rubric for decisions such as searching for company-policy questions, answering only from 
returned evidence, and declining when evidence is insufficient. The agent adapter should expose this trace so that a 
correct-looking answer cannot hide an incorrect tool choice or failed execution.

The planned end-to-end test case flow is:

1. Load a known scenario and run the real application agent.
2. Record retrieved contexts, sources, answer text, and tool trace.
3. Run the metrics appropriate to the layer and compare them with configured thresholds.
4. Investigate failures in layer order: integrity → retrieval → answer → tool/decision behavior.

Layers 1 and 2 are implemented as the foundation of this workflow. Layers 3 and 4 can be added incrementally without
replacing the real application path.
