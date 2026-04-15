---
title: PlantDocAI
emoji: "🌾"
colorFrom: green
colorTo: teal
sdk: streamlit
app_file: app.py
pinned: false
---

# PlantDocAI

PlantDocAI is a farmer-focused RAG assistant for crop disease diagnosis, pest management, and practical treatment guidance. It combines Groq LLM responses with FAISS retrieval over agriculture PDFs.

## What This Project Does

- Answers farming questions using domain PDF knowledge first (RAG mode).
- Falls back safely when retrieval is unavailable.
- Exposes:
- Streamlit app UI in [app.py](app.py)
- FastAPI endpoint in [main.py](main.py)

## Tech Stack

- Python 3.11
- LangChain + FAISS
- HuggingFace sentence embeddings (`all-MiniLM-L6-v2`)
- Groq LLM (`llama-3.3-70b-versatile`)

## Repository Structure

- [generator.py](generator.py): core response generation and mode routing
- [retriever.py](retriever.py): FAISS retriever setup
- [evaluation.py](evaluation.py): modular evaluation pipeline
- [rebuild_index.py](rebuild_index.py): rebuild FAISS from PDFs
- [file_path](file_path): agriculture source PDFs (Git LFS)
- [faiss_index](faiss_index): vector index (Git LFS)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add environment variable:

- `RAG_API_KEY=<your_groq_key>`

3. Rebuild index (if needed):

```bash
python rebuild_index.py
```

4. Run Streamlit app:

```bash
streamlit run app.py
```

5. Run API:

```bash
uvicorn main:app --reload
```

## Evaluation

Run:

```bash
python evaluation.py
```

This generates both:
- [evaluation_results.json](evaluation_results.json)
- [results.json](results.json)

### Current strong baseline (after index rebuild)

- Queries evaluated: 13
- Successful: 13
- Mode distribution: all RAG
- Average precision@K: 1.0
- Average recall@K: 0.723
- Average relevance: 0.738
- Average final score: 0.838

## Notes for Stable Deployments

- Keep PDFs and FAISS artifacts tracked via Git LFS.
- If retrieval fails with `Index type 0x73726576 ("vers") not recognized`, rebuild index and verify LFS files are hydrated.
- Do not commit `.env`.
