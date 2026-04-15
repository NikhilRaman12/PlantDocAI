import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Load Environment ─────────────────────────────
load_dotenv()


class ResponseGenerator:
    """
    Krishi Seva AI – Bharat
    Professional AI-powered crop management assistant using RAG.
    """

    def __init__(self, llm_model_name="llama-3.3-70b-versatile", temperature=0.1):
        self.last_mode = "INIT"

        # Refresh env values per instance so key rotation is picked up after restart.
        self.groq_key = os.getenv("RAG_API_KEY") or os.getenv("GROQ_API_KEY")
        self.hf_key = os.getenv("HF_API_KEY")

        if self.hf_key:
            os.environ["HF_TOKEN"] = self.hf_key

        if not self.groq_key:
            raise ValueError("Groq API key missing. Set RAG_API_KEY (or GROQ_API_KEY) in .env.")

        # ── LLM ─────────────────────────────
        self.llm = ChatGroq(
            model=llm_model_name,
            temperature=temperature,
            groq_api_key=self.groq_key
        )

        # ── Embeddings ──────────────────────
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print("Embedding load failed:", e)
            self.embeddings = None

        # ── Vectorstore & Retriever ─────────
        try:
            if self.embeddings:
                self.vectorstore = FAISS.load_local(
                    "faiss_index",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            else:
                self.vectorstore = None
                self.retriever = None
        except Exception as e:
            print("Retriever init failed:", e)
            self.vectorstore = None
            self.retriever = None

        # ── Strict RAG Prompt ───────────────
        self.rag_prompt = PromptTemplate.from_template(
            """
            SYSTEM ROLE: You are Krishi Seva AI – Bharat, a professional agricultural assistant.

            GUARDRAILS:
            - Use ONLY the provided context
            - Do NOT invent or hallucinate
            - If context is missing → reply "information not available"
            - Reject unsafe, harmful, or irrelevant queries

            FORMATTING RULES:
            - For symptoms, treatments, or management → ALWAYS use markdown table format
            - For simple factual answers → 2-3 concise sentences
            - Support multilingual queries

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

    def retrieve_documents(self, query: str) -> Optional[List[str]]:
        if not self.retriever:
            return None

        try:
            docs = self.retriever.invoke(query)
        except Exception as e:
            print("Retriever failed:", e)
            return None

        if not docs:
            return None

        filtered = [
            doc.page_content.strip()
            for doc in docs
            if hasattr(doc, "page_content") and doc.page_content and len(doc.page_content.strip()) > 20
        ]
        return filtered if filtered else None

    # ── Retrieval ─────────────────────────────
    def get_context(self, query):
        docs = self.retrieve_documents(query)
        if not docs:
            return None
        return "\n\n".join(docs)

    # ── Main Execution ───────────────────────
    def run(self, query):
        context = self.get_context(query)

        try:
            if context and len(context.strip()) > 100:
                self.last_mode = "RAG"
                print("MODE: RAG")
                chain = self.rag_prompt | self.llm
                response = chain.invoke({"context": context, "question": query})
                return type("obj", (object,), {"content": response.content, "mode": self.last_mode})

            if context:
                self.last_mode = "HYBRID"
                print("MODE: HYBRID")
                hybrid_prompt = f"""
                SYSTEM ROLE: You are Krishi Seva AI – Bharat, a professional agricultural expert.

                RULES:
                - Use context FIRST, then external knowledge ONLY if needed
                - Block unsafe or irrelevant queries
                - Follow formatting rules strictly

                Context:
                {context}

                Question:
                {query}

                Answer:
                """
                response = self.llm.invoke(hybrid_prompt)
                return type("obj", (object,), {"content": response.content, "mode": self.last_mode})

            self.last_mode = "LLM_FALLBACK"
            print("MODE: LLM FALLBACK")
            fallback_prompt = f"""
            SYSTEM ROLE: You are Krishi Seva AI – Bharat, a professional agricultural expert.

            RULES:
            - If unsure → reply "information not available"
            - Block unsafe or irrelevant queries
            - Follow formatting rules strictly

            Question:
            {query}

            Answer:
            """
            response = self.llm.invoke(fallback_prompt)
            return type("obj", (object,), {"content": response.content, "mode": self.last_mode})
        except Exception as e:
            msg = str(e)
            if "invalid_api_key" in msg or "401" in msg:
                raise RuntimeError(
                    "Invalid Groq API key (401). Update RAG_API_KEY in .env and restart the app."
                ) from e
            raise

    def generate_response(self, question, docs=None):
        """Compatibility wrapper for API callers expecting plain string output."""
        if docs and str(docs).strip():
            chain = self.rag_prompt | self.llm
            response = chain.invoke({"context": str(docs), "question": question})
            self.last_mode = "RAG"
            return response.content

        result = self.run(question)
        return getattr(result, "content", str(result))


# ── Test ─────────────────────────────
if __name__ == "__main__":
    generator = ResponseGenerator()
    query = "Symptoms of Fusarium wilt in tomato"
    result = generator.run(query)
    print(result.content)
