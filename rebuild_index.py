import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_DIR = "file_path"

# Load all PDFs from file_path/
documents = []
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs: {pdf_files}")

for pdf_file in pdf_files:
    path = os.path.join(PDF_DIR, pdf_file)
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded {len(docs)} pages from {pdf_file}")
    except Exception as e:
        print(f"  WARN: could not load {pdf_file}: {e}")

print(f"\nTotal pages loaded: {len(documents)}")

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build and save FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"\nIndex rebuilt successfully with {len(chunks)} chunks → faiss_index/")