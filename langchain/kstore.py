
from typing import List
import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import MergerRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from sentence_transformers import util

from tqdm import tqdm


class BH2V_KnowledgeStore:

    CHROMADB_PATH = "./traceability_chromadb"
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 20

    def __init__(self, create=False, path=None) -> None:

        self.db_path = path if path is not None else self.CHROMADB_PATH

        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBEDDINGS_MODEL)
        self.persistent_client = chromadb.PersistentClient(path=self.db_path, settings=Settings(allow_reset=True))

        self.embeddings = SentenceTransformerEmbeddings(model_name=BH2V_KnowledgeStore.EMBEDDINGS_MODEL) # Local embeddings

        print(create)
        if create:
            print("Deleting existing collections ...")
            self.persistent_client.reset()
        
        self.blockchain_collection = self.persistent_client.get_or_create_collection("blockchain", embedding_function=self.sentence_transformer_ef)
        self.hydrogen_collection = self.persistent_client.get_or_create_collection("green_hydrogen", embedding_function=self.sentence_transformer_ef)

        self.blockchain_vectordb = Chroma(collection_name="blockchain", persist_directory=self.db_path, embedding_function=self.embeddings)  
        print("There are", self.blockchain_vectordb._collection.count(), "chunks in the blockchain collection")

        self.hydrogen_vectordb = Chroma(collection_name="green_hydrogen", persist_directory=self.db_path, embedding_function=self.embeddings)  
        print("There are", self.hydrogen_vectordb._collection.count(), "chunks in the green hydrogen collection")
   
    @staticmethod
    def _process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in tqdm(pdf_files, "PDFs"):
            pdf_loader = PyPDFLoader(pdf_file_path)
            batch_docs.extend(pdf_loader.load())

        # text_splitter = RecursiveCharacterTextSplitter(HuggingFaceEmbeddings(model_name=KnowledgeStore.EMBEDDINGS_MODEL))
        text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name=BH2V_KnowledgeStore.EMBEDDINGS_MODEL))
        pdf_chunks = text_splitter.split_documents(batch_docs)

        return pdf_chunks
    
    def ingest_blockchain(self, pdf_files:List[str]):
        print("Ingesting PDFs for blockchain ...")
        pdf_chunks = BH2V_KnowledgeStore._process_pdf_batch(pdf_files)
        print(len(pdf_chunks), "chunks")

        if len(pdf_chunks) > 0:
            self.blockchain_vectordb = Chroma.from_documents(pdf_chunks, embedding=self.embeddings, persist_directory=self.db_path, collection_name="blockchain")
    
    def ingest_green_hydrogen(self, pdf_files:List[str]):
        print("Ingesting PDFs for green hydrogen ...")
        pdf_chunks = BH2V_KnowledgeStore._process_pdf_batch(pdf_files)
        print(len(pdf_chunks), "chunks")

        if len(pdf_chunks) > 0:
            self.hydrogen_vectordb = Chroma.from_documents(pdf_chunks, embedding=self.embeddings, persist_directory=self.db_path, collection_name="green_hydrogen")

    def search(self, query:str, collection:str, k:int=3):
        if collection == "blockchain":
            vectordb = self.blockchain_vectordb
        elif collection == "green_hydrogen":
            vectordb = self.hydrogen_vectordb
        else:
            print("Error: collection not found ...", collection)
            return []

        docs = vectordb.similarity_search(query, k=k) # It relies on Langchain wrapper
        return docs
    
    def _get_database(self, collection:str):
        if collection == "blockchain":
            vectordb = self.blockchain_vectordb
        elif collection == "green_hydrogen":
            vectordb = self.hydrogen_vectordb
        else:
            print("Error: collection not found ...", collection)
            return None
        
        return vectordb
    
    def get_retriever(self, collection:str=None, threshold:float=0.5, k: int=3):
        retriever = None
        if collection is None:
            print("Warning: collection not found ...", collection)
            return None
        
        if collection == 'all':
            print("Creating LOTR ...")
            bb_retriever = self.get_retriever('blockchain', threshold)
            gh_retriever = self.get_retriever('green_hydrogen', threshold)
            retriever = MergerRetriever(retrievers=[bb_retriever, gh_retriever])
        else:
            chroma_db = self._get_database(collection=collection)
            # retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K}) 
            retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": threshold, "k": k}) 
        
        return retriever

