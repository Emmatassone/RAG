# Getting Started with Create React App
## How to Run the Server

To run the backend server, follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required dependencies by running:
    ```bash
    pip install flask flask-cors pandas pyyaml groq
    ```
3. Navigate to the directory containing `server.py`.
4. Run the server using the following command:
    ```bash
    python server.py
    ```

The server will start on port `5000`. You can send requests to the server at `http://localhost:5000/chat`.