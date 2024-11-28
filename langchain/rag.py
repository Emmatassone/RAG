from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import RAG_PROMPT
from utils import get_context

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
PORT = 5000

# Get GROQ API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to read CSV and convert to string
def read_csv_to_string(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

# Create the prompt template
prompt = PromptTemplate.from_template(RAG_PROMPT)

# Initialize the LLM with GROQ API
llm = ChatGroq(api_key=GROQ_API_KEY)

# Create the RunnableSequence
rag_chain = prompt | llm | StrOutputParser()

# Function to interact with the user
def interact_with_user(question, table):
    context = get_context(question)
    response = rag_chain.invoke({"question": question, "table": table, "context": context})
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    table = read_csv_to_string("blockchain_table.csv")
    response = interact_with_user(user_message, table)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(port=PORT, debug=True)