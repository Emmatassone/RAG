import json
import pandas as pd
import yaml
from groq import Groq
import argparse

def load_documents(blockchain_info = 'blockchain_information.json', blockchain_table = 'blockchain_table.csv'):
    with open(blockchain_info, 'r') as f:
        dictionary = json.load(f)
        
    df = pd.read_csv(blockchain_table).to_csv(index=True)
    
    return dictionary['Info'], df


def jaccard_similarity(query, document):
    query = query.lower().split(" ")
    document = document.lower().split(" ")
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def return_response(user_input, corpus_of_documents):
    similarities = []
    for doc in corpus_of_documents:
        similarity = jaccard_similarity(user_input, doc)
        similarities.append(similarity)
    
    index = similarities.index(max(similarities))
    # print(similarities)
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
                "content": prompt.format(user_input = user_input, relevant_document = relevant_document, table = table),
            }
        ],
        model="llama3-8b-8192",
    )
    print(chat_completion.choices[0].message.content)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process user input.')
    parser.add_argument('--user_input', type=str, required=True, help='User input for the query')
    args = parser.parse_args()

    user_input = args.user_input

    blockchain_info, blockchain_table = load_documents()
    response = return_response(user_input, blockchain_info)
    print('User input:',user_input)
    print('Recommended response:', response)
    print('----------RAG ANSWER------------')
    with open('LLM_API_KEY.yml','r') as file:
        credentials = yaml.safe_load(file)
        GROQ_API_KEY = credentials['GROQ_API_KEY']

    client = Groq(api_key = GROQ_API_KEY,)
    get_groq_answer(user_input, response, blockchain_table)