from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunks import chunks   # import the chunks created in chunks.py

def build_vectorstore(chunks_list=chunks, index_path="faiss_index"):
    if not chunks_list:
        raise ValueError("No chunks found. Run data loading and chunking first.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks_list, embeddings)
    vectorstore.save_local(index_path)
    print(f"Index saved with {len(chunks_list)} chunks")
    return vectorstore


vectorstore = build_vectorstore()

if __name__ == "__main__":
    print("vectorstores.py executed directly")
