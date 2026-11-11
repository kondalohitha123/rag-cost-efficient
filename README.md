# 🤖 RAG Cost-Efficient: Local Document Assistant

A retrieval-augmented generation (RAG) app that answers questions about your own documents **entirely on your machine**: no API keys, no per-query cost, and no data leaving your computer.

Built with Python, Sentence-Transformers, Llama 3.2 (via Ollama) and Streamlit.

## Why "cost-efficient"?

| Design choice | Effect |
| --- | --- |
| Local embedding model (`all-MiniLM-L6-v2`) | No embedding API fees; runs offline |
| Local LLM (`llama3.2` via Ollama) | No per-token generation cost |
| SHA-256 hash per chunk (idempotent ingestion) | Unchanged chunks are never re-embedded, so re-running ingestion costs nothing |
| `top_k = 1` retrieval | Only the most relevant chunk goes into the prompt, keeping prompts short and fast |
| Simple JSON vector store | No vector-database service to host or pay for |

## Features

- Document ingestion with overlapping chunking (300 characters, 50 overlap)
- 384-dimensional embeddings generated locally
- Cosine-similarity retrieval implemented in pure Python
- Context-grounded answers from Llama 3.2, with a fallback to general knowledge when the context is not relevant
- Streamlit chat interface with conversation history and a "Retrieved context" panel showing the similarity score and source
- Command-line query tool (`query.py`) for quick testing

## How it works

```
 Document ──► Chunking ──► Local embeddings ──► vector_store.json
                                                      │
 Question ──► Embed question ──► Cosine similarity ───┘
                                        │
                              Top-k chunk(s) as context
                                        │
                           Llama 3.2 (Ollama) ──► Answer
```

## Project structure

```
rag-cost-efficient/
├── app.py            # Streamlit chat app
├── ingest.py         # Chunk, embed and store documents
├── query.py          # Command-line question answering
├── data/
│   └── sample.md     # Sample document (company policy)
├── requirements.txt
└── .gitignore
```

## Setup

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com) installed.

1. Clone the repository
   ```bash
   git clone https://github.com/kondalohitha123/rag-cost-efficient.git
   cd rag-cost-efficient
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download the local LLM
   ```bash
   ollama pull llama3.2
   ```
4. Build the vector store (`vector_store.json` is git-ignored, so you must create it)
   ```bash
   python ingest.py
   ```
5. Run the app
   ```bash
   streamlit run app.py
   ```

Or ask a question from the terminal:

```bash
python query.py
```

## Example

The sample document (`data/sample.md`) contains a short company policy. Try asking:

- *"What is the policy on leave and work hours?"*
- *"Which days can I work remotely?"*

## Using your own documents

Place a `.md` or `.txt` file in `data/`, point `ingest.py` at it, and run `python ingest.py` again. Only new chunks are embedded.

## Limitations and future work

- Retrieval returns only the top chunk; try a larger `TOP_K` for longer answers
- Cosine similarity runs in pure Python; NumPy or FAISS would scale better
- Only text/Markdown files are supported; PDF ingestion is a planned addition
- Add reranking and a small evaluation set to measure answer quality

## Tech stack

Python · Streamlit · Ollama (Llama 3.2) · Sentence-Transformers (`all-MiniLM-L6-v2`) · python-dotenv
