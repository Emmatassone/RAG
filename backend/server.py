from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import yaml
from groq import Groq

app = Flask(__name__)
CORS(app) 
PORT = 5000

# Test function. Not used in the final version.
def get_chatbot_response(message):
    return f"Echo: {message}"  

#/chat endpoint
def load_documents(blockchain_info='blockchain_information.json', blockchain_table='blockchain_table.csv'):
    with open(blockchain_info, 'r') as f:
        dictionary = json.load(f)
        
    df = pd.read_csv(blockchain_table).to_csv(index=True)
    
    return dictionary['Info'], df

def jaccard_similarity(query, document):
    query = query.lower().split(" ")
    document = document.lower().split(" ")
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)

def return_response(user_input, corpus_of_documents):
    similarities = []
    for doc in corpus_of_documents:
        similarity = jaccard_similarity(user_input, doc)
        similarities.append(similarity)
    
    index = similarities.index(max(similarities))
    return corpus_of_documents[index]

def get_groq_answer(user_input, relevant_document, table):
    prompt = """
    You are a bot that explain how to use a blockchain traceability platform. You answer in very short sentences and do not include extra information.
    This is the blockchain contract information: {table}
    Holder column indicates who holds the corresponding hydrogen lot, Owner column indicates who owns the hydrogen lot and the other columns indicate properties of the hydrogen lot produced by the company.
    This is the recommended answer to the user input: {relevant_document}
    The user input is: {user_input}
    Compile a recommendation to the user based on the recommended answer and the user input.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt.format(user_input=user_input, relevant_document=relevant_document, table=table),
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    blockchain_info, blockchain_table = load_documents()
    response = return_response(user_message, blockchain_info)
    
    with open('LLM_API_KEY.yml', 'r') as file:
        credentials = yaml.safe_load(file)
        GROQ_API_KEY = credentials['GROQ_API_KEY']

    global client
    client = Groq(api_key=GROQ_API_KEY)
    bot_response = get_groq_answer(user_message, response, blockchain_table)
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(port=PORT, debug=True)

