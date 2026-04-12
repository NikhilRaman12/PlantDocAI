from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class Retriever:
    blocked_words = [
        "leak", "breach", "vulnerability", "exploit", "attack",
        "compromise", "malware", "phishing", "ransomware",
        "spyware", "reveal", "ignore",
        "sensitive", "previous instructions"
    ]

    def __init__(self):
        # Same embeddings model as vectorstores.py
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Safe deserialization since you created the FAISS index yourself
        self.vector_store = FAISS.load_local(
            "faiss_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Retriever with top-k results
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, query):
        # 1. Security filter
        for word in self.blocked_words:
            if word in query.lower():
                return "query contains unsupported content"

        # 2. Use invoke (new API, avoids deprecation warning)
        docs = self.retriever.invoke(query)

        # 3. Empty safety
        if not docs:
            return None

        # 4. Filter meaningful docs
        filtered_docs = [
            doc for doc in docs
            if hasattr(doc, "page_content") and len(doc.page_content.strip()) > 20
        ]

        return filtered_docs if filtered_docs else None


if __name__ == "__main__":
    r = Retriever()
    results = r.retrieve("What is Fusarium wilt?")
    if results:
        for doc in results:
            print(doc.page_content)
    else:
        print("No documents retrieved.")
