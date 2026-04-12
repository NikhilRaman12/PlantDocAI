from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunks import chunks   # import the chunks created in chunks.py

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index from chunks
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"Index saved with {len(chunks)} chunks")

if __name__ == "__main__":
    print("vectorstores.py executed directly")
