import os
import logging
import signal
import sys
from flask import Flask, render_template, request, jsonify
import openai
import core_functions
import assistant
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ====================================================
# Configuration and Constants
# ====================================================

# Retry settings for API calls
retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(Exception)
}

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths for resources and vector store
RESOURCES_PATH = 'resources'
VECTOR_STORE_PATH = 'docs/static'

# Initialize Flask app
app = Flask(__name__)

# ====================================================
# Helper Functions
# ====================================================

def get_openai_api_key():
    """
    Retrieve the OpenAI API key from environment variables.
    Raises a ValueError if the key is not found.
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables")
    return api_key

@retry(**retry_settings)
def initialize_openai_client(api_key):
    """
    Initialize the OpenAI client with the provided API key.
    Retries on failure based on retry settings.
    """
    return openai.OpenAI(api_key=api_key)

def load_documents():
    """
    Load documents from JSON files in the specified resources path.
    """
    raw_documents = core_functions.load_json_files(RESOURCES_PATH)
    logging.info("Loaded raw documents")
    for doc in raw_documents[:5]:
        logging.debug(doc.page_content[:200])  # Preview first 200 characters
    return raw_documents

def initialize_vector_store(documents, embeddings):
    """
    Initialize the vector store with documents and embeddings, then save it locally.
    """
    vector_store = core_functions.initialize_vector_store(documents, embeddings, VECTOR_STORE_PATH)
    logging.info(f"Vector store saved locally at: {VECTOR_STORE_PATH}")
    return vector_store

def verify_vector_store(embeddings):
    """
    Load the vector store from disk and verify its integrity.
    """
    loaded_vector_store = core_functions.load_vector_store(VECTOR_STORE_PATH, embeddings)
    logging.info("Vector store loaded successfully")
    return loaded_vector_store

def setup_integrations(app, client, assistant_id):
    """
    Import and set up integrations, adding routes to the Flask app.
    """
    available_integrations = core_functions.import_integrations()
    requires_db = False

    for integration_name, integration_module in available_integrations.items():
        integration_module.setup_routes(app, client, assistant_id)
        if integration_module.requires_mapping():
            requires_db = True

    if requires_db:
        core_functions.initialize_mapping_db()
        logging.info("Database mapping initialized")

def graceful_shutdown(sig, frame):
    """
    Handle application shutdown gracefully when receiving termination signals.
    """
    logging.info("Signal received, shutting down")
    sys.exit(0)

# ====================================================
# Flask Routes
# ====================================================

@app.route('/')
def home():
    """
    Render the home page of the application.
    """
    return render_template('index.html')

# ====================================================
# Main Execution
# ====================================================

if __name__ == '__main__':
    try:
        # Retrieve API Key and initialize OpenAI client
        OPENAI_API_KEY = get_openai_api_key()
        client = initialize_openai_client(OPENAI_API_KEY)
        logging.info("OpenAI client initialized successfully")

        # Load documents and initialize embeddings
        documents = load_documents()
        embeddings = OpenAIEmbeddings()
        vector_store = initialize_vector_store(documents, embeddings)

        # Verify the vector store
        loaded_vector_store = verify_vector_store(embeddings)

        # Create or load the assistant
        assistant_id = assistant.create_assistant(client, loaded_vector_store)
        if not assistant_id:
            raise ValueError(f"No assistant found by id: {assistant_id}")

        # Set up integrations
        setup_integrations(app, client, assistant_id)

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

        # Start Flask application
        app.run(host='0.0.0.0', port=8080)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
