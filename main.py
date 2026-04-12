from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from generator import ResponseGenerator  # Import the class

# ── Initialize FastAPI app ─────────────────────────────
app = FastAPI(title="PlantDocAI API", version="1.0")

# ── Instantiate ResponseGenerator ──────────────────────
generator = ResponseGenerator()


# ── Pydantic model ─────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    docs: Optional[str] = None


# ── Root endpoint ──────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to PlantDocAI"}


# ── Dedicated POST endpoint for queries ────────────────
@app.post("/ask")
def ask(request: QueryRequest):
    """
    Accepts a question (and optional docs override) and returns an AI-generated answer.
    """
    answer = generator.generate_response(request.question, request.docs)
    return {"answer": answer}
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from generator import ResponseGenerator  # Import the class

# Initialize FastAPI app
app = FastAPI(title="PlantDocAI API", version="1.0")

# Instantiate ResponseGenerator
generator = ResponseGenerator()

# Pydantic model
class QueryRequest(BaseModel):
    question: str
    docs: Optional[str] = None

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to PlantDocAI"}

# POST endpoint for queries
@app.post("/ask")
def ask(request: QueryRequest):
    answer = generator.generate_response(request.question, request.docs)
    return {"answer": answer}
