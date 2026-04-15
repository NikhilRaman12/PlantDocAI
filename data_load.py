import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_pdfs(self) -> List:
        if not os.path.exists(self.file_path):
            print(f"Folder not found: {self.file_path}")
            return []

        all_docs = []
        files = os.listdir(self.file_path)
        print("Files found:", files)

        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.file_path, file)
                print(f"Processing: {file}")
                docs = PyPDFLoader(pdf_path).load()
                all_docs.extend(docs)
                print(f"{file} -> {len(docs)} pages loaded")

        return all_docs


if __name__ == "__main__":
    file_path = r"C:\Users\Nikhil Raman K\OneDrive\Documents\Raman_Py_Vscode\GenAi_Projects\PlantDocAi\file_path"
    loader = DataLoader(file_path)
    documents = loader.load_pdfs()
    print(f"Total pages loaded: {len(documents)}")
    if documents:
        print("Preview:", documents[0].page_content[:200])
