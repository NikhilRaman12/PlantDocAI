---
title: PlantDocAI
emoji: "🌾"
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# PlantDocAI

PlantDocAI is a **production-ready Retrieval-Augmented Generation (RAG) system** for crop disease diagnosis, pest management, and actionable agricultural guidance.

It combines **high-precision retrieval (FAISS)** with **low-latency LLM inference (Groq)** to deliver **structured, context-grounded responses for real-world farming use cases**.

---

## 🚀 Key Features

- Context-grounded answers using domain-specific agricultural PDFs  
- Adaptive pipeline: **RAG → Hybrid → Fallback**  
- Structured outputs (symptoms, diagnosis, management, spray tables)  
- Multilingual-ready design  
- Low-latency inference for real-time advisory  

---

## ⚙️ Tech Stack

- Python 3.11  
- LangChain + FAISS  
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)  
- Groq LLM (`llama-3.3-70b-versatile`)  
- Streamlit (UI) + FastAPI (API)

---

## 📂 Repository Structure

- `generator.py` → Core RAG pipeline and response generation  
- `retriever.py` → FAISS retrieval logic  
- `evaluation.py` → Modular evaluation pipeline (precision, recall, latency, cost)  
- `rebuild_index.py` → Rebuild FAISS index from PDFs  
- `file_path/` → Source agricultural PDFs (Git LFS)  
- `faiss_index/` → Vector index (Git LFS)

---

## 📊 Evaluation (Real Metrics)

- Queries evaluated: 13  
- Successful responses: 13  
- Mode distribution: 100% RAG  

**Performance:**
- Precision@K: **1.0**  
- Recall@K: **0.723**  
- Relevance: **0.738**  
- Final Score: **0.838**

These results demonstrate **high retrieval accuracy, balanced knowledge coverage, and stable real-world performance**.

---

## 🛠️ Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
Add environment variable:
RAG_API_KEY=<your_groq_key>
Rebuild index (if needed):
python rebuild_index.py
Run Streamlit app:
streamlit run app.py
Run API:
uvicorn main:app --reload
📊 Evaluation Usage

Run:

python evaluation.py

Outputs:

evaluation_results.json
results.json
⚠️ Deployment Notes
Keep PDFs and FAISS artifacts tracked via Git LFS

If retrieval fails with:

Index type 0x73726576 ("vers") not recognized

→ Rebuild index and ensure LFS files are properly pulled

Do not commit .env
💡 Project Goal

To demonstrate how GenAI can be grounded with domain knowledge to deliver reliable, scalable, and production-ready agricultural intelligence.
