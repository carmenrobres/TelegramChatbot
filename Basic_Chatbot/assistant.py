# ====================================================
# Imports
# ====================================================

import os
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import core_functions
from langsmith import traceable

# Configuration constants NOT REALLY NEEDED OR USED - 
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = "CreAI Chatbot"

# File paths
assistant_file_path = '.storage/assistant.json'
assistant_name = "CreAI"
assistant_instructions_path = 'assistant/instructions.txt'

# Retry settings for API calls
retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(Exception)
}

# ====================================================
# Helper Functions
# ====================================================

def get_assistant_instructions():
    """
    Load assistant instructions from a text file.

    Returns:
        str: The instructions for the assistant.
    """
    with open(assistant_instructions_path, 'r') as file:
        return file.read()

@retry(**retry_settings)
def create_or_update_assistant(client, assistant_data, is_update=False):
    """
    Create a new assistant or update an existing one.

    Args:
        client: The API client instance.
        assistant_data (dict): The data for the assistant.
        is_update (bool): Whether to update an existing assistant.

    Returns:
        The created or updated assistant object.
    """
    instructions = get_assistant_instructions()

    if is_update:
        # Update existing assistant
        return client.beta.assistants.update(
            assistant_id=assistant_data['assistant_id'],
            name=assistant_name,
            instructions=instructions,
            model="gpt-4o"
        )
    else:
        # Create a new assistant
        return client.beta.assistants.create(
            instructions=instructions,
            name=assistant_name,
            model="gpt-4o"
        )

# ====================================================
# Main Functions
# ====================================================

def create_assistant(client, context):
    """
    Create or load an existing assistant based on stored data.

    Args:
        client: The API client instance.
        context: The context for creating the assistant.

    Returns:
        str: The ID of the created or updated assistant.
    """
    if os.path.exists(assistant_file_path):
        with open(assistant_file_path, 'r') as file:
            assistant_data = json.load(file)
            assistant_id = assistant_data['assistant_id']

            # Generate current hash sums
            current_tool_hashsum = core_functions.generate_hashsum('tools')
            current_resource_hashsum = core_functions.generate_hashsum('resources')
            current_assistant_hashsum = core_functions.generate_hashsum('assistant.py')
            current_instructions_hashsum = core_functions.generate_hashsum('assistant/instructions.txt')

            current_assistant_data = {
                'tools_sum': current_tool_hashsum,
                'resources_sum': current_resource_hashsum,
                'assistant_sum': current_assistant_hashsum,
                'instructions_sum': current_instructions_hashsum
            }

            # Check if the assistant data is up-to-date
            if compare_assistant_data_hashes(current_assistant_data, assistant_data):
                print("Assistant is up-to-date. Loaded existing assistant ID.")
                return assistant_id
            else:
                print("Changes detected. Updating assistant...")
                try:
                    assistant = create_or_update_assistant(client, assistant_data, is_update=True)
                    updated_data = {
                        'assistant_id': assistant.id,
                        **current_assistant_data
                    }
                    save_assistant_data(updated_data, assistant_file_path)
                    print(f"Assistant (ID: {assistant_id}) updated successfully.")
                except Exception as e:
                    print(f"Error updating assistant: {e}")
    else:
        try:
            # Create a new assistant
            assistant = create_or_update_assistant(client, {}, is_update=False)
            print(f"Assistant ID: {assistant.id}")

            resource_hashsum = core_functions.generate_hashsum('resources')
            assistant_hashsum = core_functions.generate_hashsum('assistant.py')
            instructions_hashsum = core_functions.generate_hashsum('assistant/instructions.txt')

            assistant_data = {
                'assistant_id': assistant.id,
                'resources_sum': resource_hashsum,
                'assistant_sum': assistant_hashsum,
                'instructions_sum': instructions_hashsum
            }

            save_assistant_data(assistant_data, assistant_file_path)
            print(f"Assistant has been created with ID: {assistant.id}")

            assistant_id = assistant.id
        except Exception as e:
            print(f"Error creating assistant: {e}")
            raise

    return assistant_id

def save_assistant_data(assistant_data, file_path):
    """
    Save assistant data to a JSON file.

    Args:
        assistant_data (dict): The assistant data to save.
        file_path (str): The path to the file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(assistant_data, file)
    except Exception as e:
        print(f"Error saving assistant data: {e}")

def is_valid_assistant_data(assistant_data):
    """
    Check if the assistant data contains all required fields.

    Args:
        assistant_data (dict): The assistant data to validate.

    Returns:
        bool: True if all required fields are present, False otherwise.
    """
    required_keys = ['assistant_id', 'tools_sum', 'resources_sum', 'assistant_sum', 'instructions_sum']
    return all(key in assistant_data and assistant_data[key] for key in required_keys)

def compare_assistant_data_hashes(current_data, saved_data):
    """
    Compare the current assistant data hashes with the saved data hashes.

    Args:
        current_data (dict): The current assistant data hashes.
        saved_data (dict): The saved assistant data hashes.

    Returns:
        bool: True if all hashes match, False otherwise.
    """
    if not is_valid_assistant_data(saved_data):
        return False

    return (current_data['tools_sum'] == saved_data['tools_sum'] and
            current_data['resources_sum'] == saved_data['resources_sum'] and
            current_data['assistant_sum'] == saved_data['assistant_sum'] and
            current_data['instructions_sum'] == saved_data['instructions_sum'])
