# Import standard libraries
import os
import json
import hashlib
import time
import logging
import sqlite3
from datetime import datetime
import importlib.util

# Import third-party libraries
from flask import request, abort
import openai
from packaging import version
import PyPDF2
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from pprint import pprint
import json
import logging

# Download NLTK tokenizer model
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Define paths and constants
resources = 'resources'  # Path to the resources folder
mappings_db_path = '.storage/chat_mappings.db'  # Path to the SQLite database
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # OpenAI API key from environment
CUSTOM_API_KEY = os.getenv('OPENAI_API_KEY')  # Custom API key (if used)

# Retry settings for API calls (using tenacity for retry logic)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

retry_settings = {
    'stop': stop_after_attempt(3),  # Retry up to 3 attempts
    'wait': wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff between retries
    'retry': retry_if_exception_type(Exception)  # Retry only on exceptions
}

# ====================================================
# Database Initialization
# ====================================================

def initialize_mapping_db():
    """
    Initialize the SQLite database for storing chat mappings and chat logs.
    Creates two tables:
    1. chat_mappings - Stores integration, assistant, and chat details.
    2. chat_logs - Stores individual chat messages and their metadata.
    """
    conn = sqlite3.connect(mappings_db_path)
    cursor = conn.cursor()

    # Create chat_mappings table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_mappings (
            integration TEXT,
            assistant_id TEXT,
            chat_id TEXT,
            thread_id TEXT,
            date_of_creation TIMESTAMP,
            PRIMARY KEY (integration, chat_id)
        )
    ''')

    # Create chat_logs table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            chat_id TEXT,
            assistant_id TEXT,
            thread_id TEXT,
            message_id TEXT,
            message_content TEXT,
            timestamp TIMESTAMP
        )
    ''')

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

# ====================================================
# Database Operations
# ====================================================

def get_chat_mapping(integration, chat_id=None, assistant_id=None):
    """
    Retrieve chat mapping records from the database based on integration.
    Optionally filters by chat_id or assistant_id.

    Args:
        integration (str): The integration identifier.
        chat_id (str, optional): The chat identifier.
        assistant_id (str, optional): The assistant identifier.

    Returns:
        list: A list of dictionaries containing chat mapping details.
    """
    conn = sqlite3.connect(mappings_db_path)
    cursor = conn.cursor()

    # Base query for retrieving chat mappings
    query = "SELECT * FROM chat_mappings WHERE integration = ?"
    params = [integration]

    # Add filters based on provided parameters
    if chat_id:
        query += " AND chat_id = ?"
        params.append(chat_id)
    elif assistant_id:
        query += " AND assistant_id = ?"
        params.append(assistant_id)

    # Fetch the most recent 10 records
    query += " ORDER BY date_of_creation DESC LIMIT 10"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to a list of dictionaries
    return [
        dict(zip(["integration", "assistant_id", "chat_id", "thread_id", "date_of_creation"], row))
        for row in rows
    ]

def get_value_from_mapping(data, key):
    """
    Extract a specific value from the first record of the provided list of dictionaries.

    Args:
        data (list): A list of dictionaries containing mapping data.
        key (str): The key whose value needs to be retrieved.

    Returns:
        Any: The value associated with the specified key, or None if not found.
    """
    if data and isinstance(data, list) and isinstance(data[0], dict):
        return data[0].get(key)
    return None

def update_chat_mapping(integration, chat_id, assistant_id, thread_id):
    """
    Insert or update a chat mapping record in the database.

    Args:
        integration (str): The integration identifier.
        chat_id (str): The chat identifier.
        assistant_id (str): The assistant identifier.
        thread_id (str): The thread identifier.
    """
    conn = sqlite3.connect(mappings_db_path)
    cursor = conn.cursor()

    # Current timestamp for the date of creation
    date_of_creation = datetime.now()

    # Insert or update the chat mapping record
    cursor.execute(
        '''
        INSERT OR REPLACE INTO chat_mappings (integration, chat_id, assistant_id, thread_id, date_of_creation)
        VALUES (?, ?, ?, ?, ?)
        ''', (integration, chat_id, assistant_id, thread_id, date_of_creation)
    )

    conn.commit()
    conn.close()

def delete_chat_mapping(integration, chat_id):
    """
    Delete a chat mapping record from the database.

    Args:
        integration (str): The integration identifier.
        chat_id (str): The chat identifier.
    """
    conn = sqlite3.connect(mappings_db_path)
    cursor = conn.cursor()

    # Delete the specified chat mapping record
    cursor.execute(
        'DELETE FROM chat_mappings WHERE integration = ? AND chat_id = ?',
        (integration, chat_id)
    )

    conn.commit()
    conn.close()

# ====================================================
# API Key Check and OpenAI Version Check
# ====================================================

def check_api_key():
    """
    Verify the API key provided in the request headers.
    """
    api_key = request.headers.get('X-API-KEY')
    if api_key != CUSTOM_API_KEY:
        abort(401)  # Unauthorized access if API key does not match

def check_openai_version():
    """
    Ensure the installed OpenAI library version meets the required minimum version.
    """
    required_version = version.parse("1.1.1")
    current_version = version.parse(openai.__version__)

    if current_version < required_version:
        raise ValueError(
            f"Error: OpenAI version {openai.__version__} is less than the required version 1.1.1"
        )
    logging.info("OpenAI version is compatible.")

# ====================================================
# Example Function Map
# ====================================================

def function_one(arguments):
    """
    Example function that returns a result with the provided arguments.

    Args:
        arguments (dict): The arguments for the function.

    Returns:
        dict: A dictionary containing the result.
    """
    return {"result": f"Function One executed with {arguments}"}

def function_two(arguments):
    """
    Example function that returns a result with the provided arguments.

    Args:
        arguments (dict): The arguments for the function.

    Returns:
        dict: A dictionary containing the result.
    """
    return {"result": f"Function Two executed with {arguments}"}


# ====================================================
# Function Map and Tool Processing
# ====================================================

# Define a mapping of function names to their corresponding Python functions
function_map = {
    "function_one": function_one,
    "function_two": function_two
}

@retry(**retry_settings)
def process_tool_call(client, thread_id, run_id, tool_call):
    """
    Process an individual tool call by executing the mapped function.

    Args:
        client: The client instance for API communication.
        thread_id (str): The thread identifier.
        run_id (str): The run identifier.
        tool_call: The tool call object containing function name and arguments.
    """
    function_name = tool_call.function.name

    try:
        # Attempt to decode the arguments from the tool call
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e.msg}. Input: {tool_call.function.arguments}")
        arguments = {}  # Use an empty dictionary if decoding fails

    # Check if the function exists in the function map
    if function_name in function_map:
        # Retrieve the function from the function map
        function_to_call = function_map[function_name]

        # Execute the function with the provided arguments
        output = function_to_call(arguments)

        # Submit the tool output to the client API
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=[{
                "tool_call_id": tool_call.id,
                "output": json.dumps(output)  # Serialize the output to JSON
            }]
        )
        logging.info(f"Successfully processed tool call: {function_name}")
    else:
        # Log a warning if the function name is not found in the function map
        logging.warning(f"Function {function_name} not found in function_map.")

def process_tool_calls(client, thread_id, run_id):
    """
    Continuously process tool calls initiated by the assistant until completion or timeout.

    Args:
        client: The client instance for API communication.
        thread_id (str): The thread identifier.
        run_id (str): The run identifier.
    """
    start_time = time.time()  # Record the start time of the process

    while True:
        elapsed_time = time.time() - start_time

        # Break the loop if processing exceeds 20 seconds
        if elapsed_time > 20:
            logging.info("Breaking loop after 20 seconds timeout")
            break

        # Retrieve the current status of the run
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

        if run_status.status == 'completed':
            # Exit the loop if the run is completed
            break
        elif run_status.status == 'requires_action':
            # Iterate over tool calls that require action
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                try:
                    # Process each tool call
                    process_tool_call(client, thread_id, run_id, tool_call)
                except Exception as e:
                    logging.error(f"Failed to process tool call: {e}")
            time.sleep(2)  # Sleep briefly before checking again

# ====================================================
# Tool Loading from Directory
# ====================================================

def load_tools_from_directory(directory):
    """
    Dynamically load tool configurations and functions from Python files in the specified directory.

    Args:
        directory (str): The path to the directory containing Python tool modules.

    Returns:
        dict: A dictionary containing tool configurations and a function map.
    """
    tool_data = {"tool_configs": [], "function_map": {}}

    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            module_name = filename[:-3]  # Remove the .py extension
            module_path = os.path.join(directory, filename)

            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if the module has a tool configuration
            if hasattr(module, 'tool_config'):
                tool_data["tool_configs"].append(module.tool_config)

            # Map callable functions in the module to the function map
            for attr in dir(module):
                attribute = getattr(module, attr)
                if callable(attribute) and not attr.startswith("__"):
                    tool_data["function_map"][attr] = attribute

    logging.info(f"Loaded tools from directory: {directory}")
    return tool_data

# ====================================================
# Dynamic Module Import and Hash Functions
# ====================================================

def import_integrations():
    """
    Dynamically import Python modules from the 'integrations' folder.
    Modules must have a 'setup_routes' function to be included.

    Returns:
        dict: A dictionary mapping module names to the imported module objects.
    """
    directory = 'integrations'
    modules = {}

    for filename in os.listdir(directory):
        # Import only Python files that are not special files (e.g., __init__.py)
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            module_path = os.path.join(directory, filename)

            # Load the module dynamically using importlib
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Add the module to the dictionary if it has a 'setup_routes' function
            if hasattr(module, 'setup_routes'):
                modules[module_name] = module

    return modules

def generate_hashsum(path, hash_func=hashlib.sha256):
    """
    Generate a hashsum for a file or all files in a directory.

    Args:
        path (str): Path to the file or folder.
        hash_func: Hash function to use, default is sha256.

    Returns:
        str: The hexadecimal hashsum of the file or directory contents.
    """
    if not os.path.exists(path):
        raise ValueError("Path does not exist.")

    hashsum = hash_func()

    if os.path.isfile(path):
        # If it's a file, read and hash the file content
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hashsum.update(chunk)
    elif os.path.isdir(path):
        # If it's a directory, iterate through all files and hash their contents
        for subdir, _, files in os.walk(path):
            for file in sorted(files):
                filepath = os.path.join(subdir, file)
                with open(filepath, 'rb') as f:
                    while chunk := f.read(8192):
                        hashsum.update(chunk)
    else:
        raise ValueError("Path is neither a file nor a directory.")

    return hashsum.hexdigest()

def compare_checksums(checksum1, checksum2):
    """
    Compare two checksums for equality.

    Args:
        checksum1 (str): The first checksum.
        checksum2 (str): The second checksum.

    Returns:
        bool: True if the checksums are identical, False otherwise.
    """
    return checksum1 == checksum2

# ====================================================
# File Loading and Document Processing
# ====================================================

def load_json_files(folder):
    """
    Load and extract text data from JSON files in the specified folder.

    Args:
        folder (str): The path to the folder containing JSON files.

    Returns:
        list: A list of documents loaded from JSON files.
    """
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            file_path = os.path.join(folder, filename)
            # Use JSONLoader to load content from JSON files
            loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
            documents.extend(loader.load())
    return documents

def split_documents(documents, max_chunk_size=1000):
    """
    Split documents into smaller chunks using RecursiveJsonSplitter.

    Args:
        documents (list): A list of document objects.
        max_chunk_size (int): The maximum size of each chunk.

    Returns:
        list: A list of document chunks.
    """
    # Parse JSON content before splitting
    parsed_jsons = [json.loads(doc.page_content) for doc in documents]
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
    return splitter.create_documents(json_data=parsed_jsons)

def initialize_vector_store(documents, embeddings, save_path):
    """
    Initialize a vector store from the provided documents and save it locally.

    Args:
        documents (list): A list of document objects.
        embeddings: Embedding model to use for vectorization.
        save_path (str): The path to save the vector store.

    Returns:
        FAISS: The initialized FAISS vector store.
    """
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(save_path)
    return vector_store

# ====================================================
# Vector Store, Language Detection, and Text Chunking
# ====================================================

def load_vector_store(load_path, embeddings):
    """
    Load a vector store from the specified path using the provided embeddings.

    Args:
        load_path (str): The path to the saved vector store.
        embeddings: Embedding model used for deserialization.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

def detect_language(text):
    """
    Detect the language of the input text using langdetect.

    Args:
        text (str): The text whose language needs to be detected.

    Returns:
        str: The detected language code (e.g., 'en' for English), or None if detection fails.
    """
    try:
        return str(detect(text))
    except Exception as e:
        # Log the error if language detection fails
        print(f"Error detecting language: {e}")
        return None

    # Example usage:
    # text_dutch = "Hallo, waar kan ik een leuke cafe vinden in het buurt van el clot?"
    # text_catalan = "I si tinguéssim algú que pogués recopilar informació sobre el veïnat?"
    # language = detect_language(text_catalan)
    # print(f"The detected language is: {language}")

def chunk_input_message(input_text, chunk_size=3):
    """
    Tokenize the input text and split it into smaller chunks (every 3 words).

    Args:
        input_text (str): The text to be chunked.
        chunk_size (int): The number of words per chunk.

    Returns:
        list: A list of text chunks, each containing 'chunk_size' words.
    """
    # Tokenize the input text into words
    words = word_tokenize(input_text)
    
    # Create chunks of specified size
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def chunk_input_message2(input_text, chunk_size=8):
    """
    Tokenize the input text and split it into larger chunks (every 8 words).

    Args:
        input_text (str): The text to be chunked.
        chunk_size (int): The number of words per chunk.

    Returns:
        list: A list of text chunks, each containing 'chunk_size' words.
    """
    # Tokenize the input text into words
    words = word_tokenize(input_text)
    
    # Create chunks of specified size
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks