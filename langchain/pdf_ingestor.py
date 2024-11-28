import os
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFIngestor:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", db_path="./traceability_chromadb"):
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.client = PersistentClient(path = db_path, settings=Settings(allow_reset=True))
        self.vectorstore = Chroma(embedding_function=self.embeddings, client=self.client)
        self.db_path = db_path

    def ingest_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)

    def ingest_pdfs_from_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory_path, filename)
                self.ingest_pdf(pdf_path)

    def retrieve_context(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)

    def load_or_ingest(self, directory_path):
        if not os.path.exists(self.db_path):
            print("Database not found. Ingesting PDF files from directory...")
            self.ingest_pdfs_from_directory(directory_path)
        else:
            print("Loading existing database...")