import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import warnings
import pandas as pd

from kstore import BH2V_KnowledgeStore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.agents import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.prompt import PromptTemplate

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType


warnings.filterwarnings('ignore')

EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
CSV_BLOCKCHAIN_FILE = "./docs/blockchain_table.csv"

def get_retriever(llm, collection="all", threshold=0.5, k=3):
    # Composite retriever 
    retriever = kstore.get_retriever(collection=collection, threshold=threshold, k=3)

    # Re-phrase the query in order to increase the chances of getting documents
    retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    redundancy_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)

    return compression_retriever

def run_unstructured_rag(query, retriever, llm):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                           retriever=retriever, return_source_documents=True)
    result = qa_chain.invoke(query)
    return result

def unstructured_rag_tool(query: str) -> str: # Retriever and LLM are global
    result = run_unstructured_rag(query, retriever, llm)
    return result['result']


# --- Main program

ENV_PATH = sys.path[0]+'/andres.env'
print("Reading LLM config:", ENV_PATH, load_dotenv(dotenv_path=Path(ENV_PATH)))
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# print(os.environ['GROQ_API_KEY'])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print()
chroma_db = "./traceability_chromadb" 
kstore = BH2V_KnowledgeStore(path=chroma_db)

print()
# query = "How can I handle traceability of hydrogen units using smart contracts?"
query = "How to access the blockchain?"
print("Query:", query)
docs = kstore.search(query, collection='green_hydrogen')
for idx, doc in enumerate(docs):
    print(idx,"---")
    print(doc.page_content)
print("---")

print()
# Configure LLM for RAG
llm = ChatGroq(temperature=0.001, groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama3-70b-8192")
# llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini")

retriever = get_retriever(llm, threshold=0.4, k=3)

# 1. RAG using multiple unstructured sources (no memory yet)
result = run_unstructured_rag(query, retriever, llm)
print(result['result'])
print("Sources:", len(result['source_documents']), "documents (chunks)")

print()

# 2. RAG as a ReAct agent that combines a Pandas dataframe and a custom tool for unstructured sources
df = pd.read_csv(CSV_BLOCKCHAIN_FILE)
custom_tools = [
    Tool(name="unstructured", func=unstructured_rag_tool,
        description="Useful for getting relevant information for the query from a set of predefined documents about blockchain and green hydrogen"
    )
]

SYSTEM_PROMPT = """
You are an assistant that explains how to use a blockchain traceability platform. 
You answer should consist of concise sentences, without including extra information.
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question being asked."""

agent = create_pandas_dataframe_agent(llm, df, extra_tools=custom_tools,
                                      prefix=SYSTEM_PROMPT,
                                      # agent_type=AgentType.OPENAI_FUNCTIONS,
                                      verbose=True, max_iterations=7)

# query = "Based on the smart contracts deployed in the mainnet, describe and count the hydrogen assets from my portfolio that have been traced?"
query = "How many contracts are being transported?"
print(query)
# response = agent.invoke(input=query)
# print(response)

print()

RAG_PROMPT = """
You are an assistant that explains how to use a blockchain traceability platform and engages in question-answering tasks. 
Use the following pieces of retrieved context and blockchain contracts (in tabular format) to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep your answer concise, without including extra information.

Question: {question} 

Blockchain contracts: {table}

Context: {context} 

Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Configure a RAG chain that includes de dataframe (no memory yet)
prompt = PromptTemplate.from_template(RAG_PROMPT)
rag_chain = (
    {"context": retriever | format_docs, 
     "question": RunnablePassthrough(), 
     "table": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print()
response = rag_chain.invoke({"question":query, "table":df.head()})
print(response)

print()
