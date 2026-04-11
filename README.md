# RAG Document Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system built with Python, Streamlit, Qdrant Cloud, and Google Gemini. Chat with your own PDF, DOCX, and TXT documents like ChatGPT.

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT
- **Intelligent chunking** — sentence-aware splitting with overlap
- **Hybrid search** — vector similarity + BM25 keyword re-ranking
- **Embedding caching** — disk cache avoids recomputing known texts
- **Grounded generation** — LLM strictly answers from retrieved context
- **Streaming responses** — real-time token output
- **Chat memory** — conversational continuity across turns
- **Source citations** — every answer shows document + page

## Architecture

```
User Query
    ↓
Embed Query (HuggingFace / Gemini)
    ↓
Vector Search (Qdrant Cloud, cosine similarity)
    ↓
BM25 Re-ranking (hybrid keyword + semantic)
    ↓
Context Injection → Gemini LLM
    ↓
Grounded Answer + Sources
```

## Project Structure

```
Rag-Teach/
├── .env                 # API keys and config
├── requirements.txt     # Python dependencies
├── config.py            # Central configuration
├── utils.py             # Hashing, caching utilities
├── ingestion.py         # PDF/DOCX/TXT text extraction
├── chunking.py          # Sentence-aware chunking
├── embeddings.py        # HuggingFace/Gemini embeddings + cache
├── vector_store.py      # Qdrant Cloud integration
├── retriever.py         # Hybrid search + re-ranking
├── generator.py         # Gemini LLM with grounded prompt
└── app.py               # Streamlit chat UI
```

## Setup

### 1. Clone and enter the project

```cmd
cd c:\github\Rag-Teach
```

### 2. Create a virtual environment

```cmd
python -m venv venv
```

### 3. Activate the venv

**Windows CMD:**
```cmd
venv\Scripts\activate
```

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

**Git Bash:**
```bash
source venv/Scripts/activate
```

### 4. Install dependencies

```cmd
pip install -r requirements.txt
```

### 5. Configure API keys

Edit [.env](.env) and set:

```env
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
GEMINI_API_KEY=your-gemini-api-key

EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gemini-3.1-pro-preview
QDRANT_COLLECTION=rag_documents
```

**Get API keys:**
- Qdrant Cloud → https://cloud.qdrant.io
- Google Gemini → https://aistudio.google.com/apikey

## Run

```cmd
streamlit run app.py
```

Open your browser at **http://localhost:8501**

## Usage

1. **Upload documents** from the sidebar (PDF / DOCX / TXT — multiple files supported)
2. Click **Process Documents** — this extracts text, chunks it, generates embeddings, and uploads to Qdrant
3. **Ask questions** in the chat — the system retrieves relevant passages and generates a grounded answer
4. **View sources** by expanding the Sources panel under each response

## Configuration

All settings live in [.env](.env):

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant Cloud endpoint | — |
| `QDRANT_API_KEY` | Qdrant API key | — |
| `QDRANT_COLLECTION` | Collection name | `rag_documents` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `EMBEDDING_PROVIDER` | `huggingface` or `gemini` | `huggingface` |
| `EMBEDDING_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | Gemini model | `gemini-3.1-pro-preview` |

Tune chunking and retrieval in [config.py](config.py):

```python
CHUNK_SIZE = 512         # tokens per chunk
CHUNK_OVERLAP = 64       # token overlap between chunks
TOP_K = 5                # initial retrieval size
RERANK_TOP_K = 3         # final chunks after re-ranking
```

## Tech Stack

- **UI** — Streamlit
- **Vector DB** — Qdrant Cloud
- **Embeddings** — sentence-transformers (HuggingFace) or Gemini `text-embedding-004`
- **LLM** — Google Gemini (`gemini-3.1-pro-preview`)
- **Chunking** — tiktoken tokenizer
- **Re-ranking** — rank-bm25
- **Document parsing** — PyPDF2, python-docx

## Troubleshooting

**`Collection rag_documents doesn't exist`**
→ Upload and process documents first before asking questions.

**`GEMINI_API_KEY is not set`**
→ Add your key to [.env](.env) and restart the app.

**Slow first run**
→ HuggingFace downloads the embedding model on first use (~90 MB).

**Stop the app**
→ Press `Ctrl+C` in the terminal.

**Deactivate the venv later**
→ Run `deactivate`.

## License

MIT
