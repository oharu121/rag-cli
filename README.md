# RAG Practice

A simple RAG (Retrieval-Augmented Generation) implementation for learning purposes.

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | sentence-transformers | Free, local, multilingual |
| Vector DB | Chroma | Simple, persistent storage |
| LLM | Google Gemini 2.0 Flash | Free tier available |
| Framework | LangChain | Well-documented |

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up API key
cp .env.example .env
# Edit .env and add your Gemini API key from https://aistudio.google.com/apikey

# 3. Run
uv run rag
```

## Usage

1. Add `.txt` files to the `documents/` folder
2. Run the app and ask questions
3. Type `rebuild` to reload documents after adding new ones
4. Type `quit` to exit

## Project Structure

```
rag-practice/
  main.py           # RAG implementation
  pyproject.toml    # Dependencies
  .env.example      # API key template
  documents/        # Your documents go here
    sample.txt
  chroma_db/        # Vector database (auto-created)
```

## Configuration

Edit these in `main.py`:

```python
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## Requirements

- Python 3.12+
- uv package manager (https://docs.astral.sh/uv/)
- Google API key (free at https://aistudio.google.com/apikey)
