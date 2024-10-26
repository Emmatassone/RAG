from flask import Flask, request, jsonify

app = Flask(__name__)

def process_message(message):
    # Future interaction with RAG
    return f"You said: {message}"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    response = process_message(message)

    return jsonify({'response': response})

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(port=5000, debug=True)
