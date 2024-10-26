# Running the RAG.py Script

To run the `RAG.py` script, follow these steps:

1. **Ensure Dependencies are Installed**:
    Make sure you have all the required libraries installed. You can install them using pip:
    ```sh
    pip install pandas pyyaml groq
    ```

2. **Prepare the Required Files**:
    Ensure you have the following files in the same directory as `RAG.py`:
    - `blockchain_information.json`: Contains the blockchain information.
    - `blockchain_table.csv`: Contains the blockchain table data.
    - `LLM_API_KEY.yml`: Contains the API key for the Groq service. 

If you do not have a Groq API key, you can create one by visiting [https://console.groq.com/keys](https://console.groq.com/keys).

3. **Run the Script**:
    Use the command line to run the script with the required user input. Replace `<user_input>` with your actual query:
    ```sh
    python RAG.py --user_input "<user_input>"
    ```

Example:
```sh
python RAG.py --user_input "Explain the ownership of hydrogen lots"
```

This will process the input and provide a response based on the blockchain information and table data.
