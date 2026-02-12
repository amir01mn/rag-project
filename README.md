## RAG-1: Retrieval-Augmented Generation Learning Project

**rag-1** is a small, educational Retrieval-Augmented Generation (RAG) project in Python. It walks through loading documents (PDFs and text), splitting them into chunks, embedding them with `sentence-transformers`, storing them in ChromaDB, and answering questions using Groq-hosted LLMs via LangChain.

---

### Features

- **End‑to‑end RAG pipeline**: From raw documents to question answering.
- **Multiple loaders**: Load PDFs and text files from local folders using LangChain loaders.
- **Chunking with overlap**: Split documents into overlapping chunks using `RecursiveCharacterTextSplitter`.
- **Local embeddings**: Use `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings locally.
- **Persistent vector store**: Store and reuse embeddings with ChromaDB in `data/vector_store/`.
- **Groq LLM integration**: Run retrieval‑augmented QA with Groq models (e.g. `gemma2-9b-it`).
- **Notebook‑driven exploration**: Learn RAG concepts step by step in Jupyter notebooks.

---

### Tech Stack

- **Language**: Python 3.13
- **Core libraries**:
  - **LangChain / LangChain Core / LangChain Community**
  - **sentence-transformers**
  - **ChromaDB**
  - **python-dotenv**
- **LLMs**:
  - **Groq** (via `ChatGroq`)
  - **OpenAI** (optional, via `langchain-openai`)
- **Document loaders**:
  - PyPDF / PyMuPDF (via LangChain loaders)
  - `TextLoader`, `DirectoryLoader`
- **Environment & packaging**:
  - `uv` (with `uv.lock`) as primary package manager

---

### Project Structure

```text
rag-1/
  main.py                    # Simple entry point (currently prints a greeting)
  src/
    __init__.py              # Reserved for future reusable modules
  notebook/
    document.ipynb           # Basic document loading / text file examples
    pdfLoader.ipynb          # Full RAG pipeline: load → chunk → embed → store → query
  data/
    pdf/                     # Source PDFs for RAG (survey papers, research PDFs, etc.)
    text_files/              # Example text files (e.g. python_intro, machine_learning)
    vector_store/            # ChromaDB persistent storage (SQLite + index files)
  uv.lock                    # Dependency lockfile for uv
  .env                       # Environment variables (not committed; see below)
```

- **`main.py`**: Minimal script with a `main()` function that currently just prints `"Hello from rag-1!"`. You can later extend this to run a CLI or API server.
- **`notebook/document.ipynb`**:
  - Shows how to use `Document`, `TextLoader`, and `DirectoryLoader`.
  - Demonstrates creating and saving text files (e.g. `python_intro.txt`, `machine_learning.txt`) into `data/text_files/`.
- **`notebook/pdfLoader.ipynb`**:
  - Loads PDFs from `data/pdf/` using loaders like `PyMuPDFLoader`.
  - Splits documents using `RecursiveCharacterTextSplitter` (chunk size ~1000, overlap ~200).
  - Uses a custom `EmbeddingManager` (based on `sentence-transformers`) to embed chunks.
  - Uses a `VectorStore` abstraction around ChromaDB’s `PersistentClient` to store and retrieve embeddings.
  - Builds a simple RAG QA loop with Groq LLMs.

---

### Installation

#### 1. Clone the repository

```bash
git clone <your-github-url>/rag-learn.git
cd rag-learn/rag-1
```

#### 2. Create and activate a virtual environment (optional but recommended)

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

#### 3. Install dependencies

If you use **uv** (recommended):

```bash
uv sync
```

Or, if you prefer **pip** (assuming a `requirements.txt` has been exported):

```bash
pip install -r requirements.txt
```

---

### Configuration

The project uses environment variables for API keys, typically loaded from a `.env` file (via `python-dotenv`).

Create a `.env` file in the `rag-1/` root with at least:

```bash
GROQ_API_KEY=your_groq_api_key_here

# Optional: if you want to use OpenAI models as well
OPENAI_API_KEY=your_openai_api_key_here
```

- **`GROQ_API_KEY`**: Required for Groq LLMs used in the RAG notebook.
- **`OPENAI_API_KEY`**: Optional; used only if you select OpenAI models in your experiments.
- Ensure `.env` is **not** committed to version control.

---

### Usage

#### 1. Basic script entry point

From the `rag-1` directory:

```bash
python main.py
```

This currently serves as a placeholder entry point (prints a greeting). You can extend `main.py` to:

- Trigger document ingestion and indexing.
- Expose a CLI for asking questions.
- Start a web server or API.

#### 2. Running the RAG notebooks

1. Start Jupyter from the project root:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Open:

   - `notebook/document.ipynb` to explore:
     - Loading text files.
     - Creating and saving simple example documents in `data/text_files/`.
   - `notebook/pdfLoader.ipynb` to walk through the full RAG pipeline:
     - Load PDFs from `data/pdf/`.
     - Split into chunks using `RecursiveCharacterTextSplitter`.
     - Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
     - Persist embeddings in ChromaDB (`data/vector_store/`).
     - Retrieve relevant chunks for a user question.
     - Run a Groq LLM to generate an answer grounded in retrieved context.

3. Adjust parameters (chunk size, overlap, model names, etc.) directly in the notebook cells to experiment with different RAG configurations.

---

### Data & Vector Store

- **Source PDFs**: Place your documents in `data/pdf/`. The notebooks assume this folder exists and contains PDFs.
- **Text examples**: `notebook/document.ipynb` can create and load text files under `data/text_files/`.
- **Vector store**:
  - Uses **ChromaDB** with a persistent client.
  - Files (e.g. `chroma.sqlite3` and index files) are stored under `data/vector_store/`.
  - Once built, the vector store can be reused across notebook sessions without recomputing embeddings.

---

### Extending the Project

- **Make `main.py` useful**:
  - Wire in the components from the notebooks (loaders, splitter, embedding manager, vector store, and QA chain).
  - Add CLI options (e.g. `--index`, `--ask "<question>"`).
- **Move logic into `src/`**:
  - Extract classes such as `EmbeddingManager`, `VectorStore`, and RAG chains from notebooks into modules under `src/`.
  - Import them in notebooks and scripts for cleaner structure.
- **Add a web/API layer**:
  - Use FastAPI, Flask, or another framework to expose a `/ask` endpoint powered by the RAG pipeline.

---

### Testing & Tooling

- **Tests**: There is currently no formal test suite (e.g. pytest) configured.
- **Linting / formatting**: No explicit configuration for tools like Ruff or Black has been added yet.
- **Recommended next steps**:
  - Add `pytest` tests for key components (embedding, vector store, retrieval).
  - Introduce formatting and linting (e.g. Black, Ruff) and optionally a `pre-commit` configuration.

---

### Who Is This For?

- **Learners** who want a concrete, notebook-driven example of RAG.
- **Researchers and tinkerers** exploring different document types, chunking strategies, and LLMs.
- **Developers** who want a minimal, extensible skeleton to build a more complete RAG application.



