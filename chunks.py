from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_load import DataLoader

DATA_PATH = r"C:\Users\Nikhil Raman K\OneDrive\Documents\Raman_Py_Vscode\GenAi_Projects\PlantDocAi\file_path"


def build_chunks(data_path=DATA_PATH, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    loader = DataLoader(data_path)
    docs = loader.load_pdfs()
    return splitter.split_documents(docs)


chunks = build_chunks()
print("Total chunks:", len(chunks))
if chunks:
    print("Preview:", chunks[0].page_content[:200])

# Allow running standalone
if __name__ == "__main__":
    print("Chunks.py executed directly")
    print("Total chunks:", len(chunks))
    if chunks:
        print("First chunk preview:", chunks[0].page_content[:200])
