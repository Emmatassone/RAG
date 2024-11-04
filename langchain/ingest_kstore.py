
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from kstore import BH2V_KnowledgeStore

# warnings.filterwarnings('ignore')

BLOCKCHAIN_DOCS_PATH = "./docs/blockchain"
GREEN_HYDROGEN_DOCS_PATH = "./docs/green-hydrogen"

def create_database(folder=None):
    kstore = BH2V_KnowledgeStore(create=True, path=folder)

    # Ingestion of blockchain
    pdf_files_to_process = []
    for root, dirs, files in os.walk(BLOCKCHAIN_DOCS_PATH):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])
    for file in pdf_files_to_process:
        print(file)
    print(len(pdf_files_to_process), "pdfs to process")
    kstore.ingest_blockchain(pdf_files_to_process)
    
    # Ingestion of green hydrogen
    pdf_files_to_process = []
    for root, dirs, files in os.walk(GREEN_HYDROGEN_DOCS_PATH):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])
    for file in pdf_files_to_process:
        print(file)
    print(len(pdf_files_to_process), "pdfs to process")
    kstore.ingest_green_hydrogen(pdf_files_to_process)

    return kstore


# --- Main program

print()
kstore = create_database(folder="./traceability_chromadb")
print()