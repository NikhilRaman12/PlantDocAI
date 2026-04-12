import os
from dotenv import load_dotenv

from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from retriever import Retriever

# ── Load Environment ─────────────────────────────
load_dotenv()
api_key = os.getenv("RAG_API_KEY")


class ResponseGenerator:
    """
    Krishi Seva AI – Bharat
    Professional AI-powered crop management assistant using RAG.
    """

    def __init__(self, llm_model_name="llama-3.3-70b-versatile", temperature=0.1):
        # ── LLM ─────────────────────────────
        self.llm = ChatGroq(
            model=llm_model_name,
            temperature=temperature,
            groq_api_key=api_key
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
                self.retriever = Retriever(self.vectorstore)
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

            META-FILTERS:
            - Agricultural domain only (crops, soil, pests, diseases, management)
            - Block queries about politics, personal health, or unrelated topics

            FORMATTING RULES:
            - For symptoms, treatments, or management → ALWAYS use markdown table format
            - For simple factual answers → 2-3 concise sentences
            - Support multilingual queries

            EXAMPLE TABLE FORMAT:
            | Aspect   | Details |
            |----------|---------|
            | Symptom 1 | Description |
            | Symptom 2 | Description |

            Context:
            {context}

            Question:
            {question}

            Answer (use table format if question asks about symptoms, treatments, or management):
            """
        )

    # ── Retrieval ─────────────────────────────
    def get_context(self, query):
        if not self.retriever:
            return None

        docs = self.retriever.retrieve(query)

        if isinstance(docs, str):
            return docs

        if docs and len(docs) > 0:
            return "\n\n".join([doc.page_content for doc in docs])

        return None

    # ── Main Execution ───────────────────────
    def run(self, query):
        # STEP 1: Retrieval
        context = self.get_context(query)

        # Guardrail: Blocked query
        if context == "query contains unsupported content":
            return type("obj", (object,), {
                "content": "query contains unsupported content"
            })

        # STEP 2: Strong Context → Pure RAG
        if context and len(context.strip()) > 100:
            print("MODE: RAG")
            chain = self.rag_prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": query
            })
            return type("obj", (object,), {"content": response.content})

        # STEP 3: Weak Context → Hybrid
        if context:
            print("MODE: HYBRID")
            hybrid_prompt = f"""
            SYSTEM ROLE: You are Krishi Seva AI – Bharat, a professional agricultural expert.

            RULES:
            - Use context FIRST, then external knowledge ONLY if needed
            - Block unsafe or irrelevant queries
            - Follow formatting rules strictly

            FORMATTING RULES:
            - For symptoms, treatments, management → Markdown table
            - For simple questions → 2-3 sentences
            - Support multilingual queries

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
            response = self.llm.invoke(hybrid_prompt)
            return type("obj", (object,), {"content": response.content})

        # STEP 4: No Context → Fallback
        print("MODE: LLM FALLBACK")
        fallback_prompt = f"""
        SYSTEM ROLE: You are Krishi Seva AI – Bharat, a professional agricultural expert.

        RULES:
        - If unsure → reply "information not available"
        - Block unsafe or irrelevant queries
        - Follow formatting rules strictly

        FORMATTING RULES:
        - For symptoms, treatments, management → Markdown table
        - For simple questions → 2-3 sentences
        - Support multilingual queries

        Question:
        {query}

        Answer:
        """
        response = self.llm.invoke(fallback_prompt)
        return type("obj", (object,), {"content": response.content})


# ── Test ─────────────────────────────
if __name__ == "__main__":
    generator = ResponseGenerator()
    query = "Symptoms of Fusarium wilt in tomato"
    result = generator.run(query)
    print(result.content)
    
