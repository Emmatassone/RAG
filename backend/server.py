from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
PORT = 5000

# Sample response. This message will be processed by the RAG modules and an then get an answer
def get_chatbot_response(message):
    return f"Echo: {message}"  

#/chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    bot_response = get_chatbot_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(port=PORT, debug=True)

