import os
from langchain_community.document_loaders import PyPDFLoader

# Use the actual folder where your PDFs are stored
file_path = r"C:\Users\Nikhil Raman K\OneDrive\Documents\Raman_Py_Vscode\GenAi_Projects\PlantDocAi\file_path"

print("Checking folder:", file_path)

if not os.path.exists(file_path):
    print("Folder NOT found")
else:
    files = os.listdir(file_path)
    print("Files found:", files)

    for file in files:
        if file.lower().endswith(".pdf"):
            print(f"Processing: {file}")

            pdf_path = os.path.join(file_path, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            print(f"{file} -> {len(docs)} pages loaded")

            if docs:
                # Preview the first 200 characters of the first page
                print("Preview:", docs[0].page_content[:200])

            print("-" * 40)
