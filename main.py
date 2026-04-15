from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from generator import ResponseGenerator

app = FastAPI(title="PlantDocAI API", version="1.0")
generator = ResponseGenerator()


class QueryRequest(BaseModel):
    question: str
    docs: Optional[str] = None


@app.get("/")
def root():
    return {"message": "Welcome to PlantDocAI"}


@app.post("/ask")
def ask(request: QueryRequest):
    answer = generator.generate_response(request.question, request.docs)
    return {"answer": answer, "mode": generator.last_mode}
