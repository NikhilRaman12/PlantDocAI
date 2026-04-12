from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_load import DataLoader

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Load documents from file_path using DataLoader
loader = DataLoader(
    r"C:\Users\Nikhil Raman K\OneDrive\Documents\Raman_Py_Vscode\GenAi_Projects\PlantDocAi\file_path"
)
docs = loader.load_pdfs()

# Split into chunks
chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))

if chunks:
    print("Preview:", chunks[0].page_content[:200])

# Allow running standalone
if __name__ == "__main__":
    print("Chunks.py executed directly")
    print("Total chunks:", len(chunks))
    if chunks:
        print("First chunk preview:", chunks[0].page_content[:200])
