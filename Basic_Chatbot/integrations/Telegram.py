
# ====================================================
# Imports
# ====================================================

import os
import logging
import telebot
import core_functions
from openai import OpenAI
from pathlib import Path
from pprint import pprint
from openai import NotFoundError
from langdetect import detect
from dotenv import load_dotenv
import tempfile
from threading import Thread, Event, current_thread
from flask import Flask, request, jsonify
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
import nltk
from nltk.tokenize import word_tokenize
# translator
from deep_translator import GoogleTranslator
#for image recognition
import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
#for audio transcription
import soundfile as sf
import transcribe
from langsmith import traceable
from sheets_client import append_to_sheet1
from datetime import datetime

# Load environment variables
load_dotenv()

# Download the NLTK tokenizer model
nltk.download('punkt')

# Configuration
vector_store_path = 'docs/static'
embeddings = OpenAIEmbeddings()
language_status = None
messageEN = None
telegram_chat_id = None
stop_event = Event()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Retry settings for API calls
retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(requests.exceptions.RequestException)
}

# ====================================================
# Audio Transcription Function
# ====================================================

def transcribe_audio(audio_file_path, api_key):
    """
    Transcribe an audio file using OpenAI Whisper API.

    Args:
        audio_file_path (str): Path to the audio file.
        api_key (str): OpenAI API key.

    Returns:
        str: Transcribed text or None if transcription fails.
    """
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        transcription_text = response.text
        if not transcription_text:
            logging.error("Transcription result is empty.")
            return None
        return transcription_text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

# ====================================================
# Setup Telegram Bot and Routes
# ====================================================

def requires_mapping():
    """
    Specify whether a database mapping is required for the bot.
    
    Returns:
        bool: True indicating that mapping is required.
    """
    return True

@traceable
def setup_routes(app, client, assistant_id):
    """
    Set up the Telegram bot routes and handlers.

    Args:
        app: Flask app instance.
        client: OpenAI client instance.
        assistant_id (str): ID of the assistant.
    """
    global language_status, messageEN, telegram_chat_id

    TELEGRAM_TOKEN = os.getenv('TELEGRAM_API_KEY')
    if not TELEGRAM_TOKEN:
        raise ValueError("No Telegram token found in environment variables")

    global bot
    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    greetings = [
        'hola', 'Hola', 'hola creai', 'hola creAi', 'hi', 'Hi!',
        'hi there!', 'hey', 'greetings', 'hola', 'bonjour'
    ]

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        """
        Handle the /start command and initiate a conversation.
        """
        telegram_chat_id = message.chat.id
        user = message.from_user
        logging.info("Starting a new conversation...")

        chat_mapping = core_functions.get_chat_mapping("telegram", telegram_chat_id, assistant_id)
        if not chat_mapping:
            thread = client.beta.threads.create()
            core_functions.update_chat_mapping("telegram", telegram_chat_id, assistant_id, thread.id)
            logging.info(f"New thread created with ID: {thread.id}")

        welcome_message = f"Hola {user.first_name}! ✨ Sóc CreAI"
        bot.reply_to(message, welcome_message)

    @bot.message_handler(func=lambda message: message.text.lower() in greetings)
    def greet_user(message):
        """
        Respond to user greetings.
        """
        telegram_chat_id = message.chat.id
        user = message.from_user
        welcome_message = f"Hola {user.first_name}! Com estás?"
        bot.reply_to(message, welcome_message)

    @bot.message_handler(content_types=['voice'])
    def handle_audio(message):
        """
        Handle voice messages, convert and transcribe audio, and process the response.
        """
        telegram_chat_id = message.chat.id
        openai_api_key = os.getenv('OPENAI_API_KEY')

        # Create temporary files for OGG and WAV formats
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_ogg:
            temp_ogg_path = temp_ogg.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            temp_wav_path = temp_wav.name

        try:
            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            # Save the OGG file
            with open(temp_ogg_path, 'wb') as f:
                f.write(downloaded_file)

            # Convert OGG to WAV
            convert_ogg_to_wav(temp_ogg_path, temp_wav_path)

            # Transcribe the audio
            logging.info("Processing transcription...")
            result_transcription = transcribe_audio(temp_wav_path, openai_api_key)

            if result_transcription:
                logging.info(f"Transcription: {result_transcription}")
                user_input = str(result_transcription)
            else:
                bot.send_message(telegram_chat_id, "Could not transcribe the audio.")
                os.remove(temp_ogg_path)
                os.remove(temp_wav_path)
                return

            os.remove(temp_ogg_path)
            os.remove(temp_wav_path)

            # Process the transcription for further actions
            messageEN = user_input
            logging.info(f"Message for GPT-3: {messageEN}")

            # Tokenize and chunk the input message
            chunks = core_functions.chunk_input_message2(messageEN)
            logging.info(f"Chunks: {chunks}")

            # Echo the response based on the chunks
            echo_all(telegram_chat_id, messageEN, chunks)

        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            bot.send_message(telegram_chat_id, "An error occurred while processing the audio.")

# ====================================================
# Telegram Bot Handlers for Photo and Text Messages
# ====================================================

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        """
        Handle photo messages sent by the user.

        Args:
            message: The Telegram message object containing the photo.
        """
        telegram_chat_id = message.chat.id

        # Get the file ID of the largest photo size and download the image
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"

        # Get additional text input (caption) with the photo and translate it
        user_message = message.caption
        if user_message:
            user_message = GoogleTranslator(source='auto', target='en').translate(text=user_message)
        else:
            user_message = ""

        try:
            # Download and encode the image in base64
            image_data = base64.b64encode(download_file(image_url)).decode("utf-8")
            logging.info("Image data encoded successfully.")

            # Initialize the OpenAI model
            model = ChatOpenAI(model="gpt-4o")

            # Create a message for the model with the image and a request for description
            ai_message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Describe the contents of this image. Please keep the answer short, factual, and based purely on the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ])

            # Invoke the model to get the description of the image
            response = model.invoke([ai_message])
            logging.info(f"Image description: {response.content}")

            # Combine the user message (caption) and the AI-generated description
            user_input = f"{user_message}\n\n{response.content}"

            # Detect the language of the combined input
            language = core_functions.detect_language(user_input)
            logging.info(f"Detected Language: {language}")
            global language_status
            language_status = language

            # Translate the combined input to English if necessary
            messageEN = GoogleTranslator(source='auto', target='en').translate(text=user_input)
            logging.info(f"English Message: {messageEN}")

            # Tokenize and chunk the input message
            chunks = core_functions.chunk_input_message2(messageEN)
            logging.info(f"Message chunks: {chunks}")

            # Proceed with common processing logic
            echo_all(telegram_chat_id, messageEN, chunks)

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download image: {e}")
            bot.send_message(telegram_chat_id, "Failed to process the image. Please try again later.")
        except Exception as e:
            logging.error(f"Error processing photo message: {e}")
            bot.send_message(telegram_chat_id, "An unexpected error occurred while processing the image.")

    @bot.message_handler(func=lambda message: True)
    def handle_text(message):
        """
        Handle text messages sent by the user.

        Args:
            message: The Telegram message object containing the text.
        """
        telegram_chat_id = message.chat.id
        user_input = message.text
        logging.info(f"Received message: {user_input}")

        # For now, we are not translating the input message to English
        messageEN = user_input
        logging.info(f"English Message: {messageEN}")

        # Tokenize and chunk the input message
        chunks = []
        chunks = core_functions.chunk_input_message(messageEN)
        logging.info(f"Message chunks: {chunks}")

        # Proceed with common processing logic
        echo_all(telegram_chat_id, messageEN, chunks)

# ====================================================
# Echo Handler for All Messages
# ====================================================

    @bot.message_handler(func=lambda message: True)
    def echo_all(telegram_chat_id, messageEN, chunk):
        """
        Handle and process all incoming messages, retrieve relevant information, 
        and send the response using OpenAI.

        Args:
            telegram_chat_id (int): The Telegram chat ID.
            messageEN (str): The English-translated message from the user.
            chunk (list): List of tokenized chunks of the input message.
        """
        # Use the provided chunk for processing
        chunks = []
        chunks = chunk

        # Load the vector store for retrieving relevant documents
        relevant_info = core_functions.load_vector_store(vector_store_path, embeddings)
        all_retrieved_docs = []

        # Retrieve the most relevant documents based on the input chunks
        for chunk in chunks:
            retriever = relevant_info.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            retrieved_docs = retriever.invoke(chunk)
            all_retrieved_docs.extend(retrieved_docs)

        # Remove duplicate documents
        all_retrieved_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
        logging.info(f"Number of retrieved documents: {len(all_retrieved_docs)}")

        # Get the thread ID from the database or create a new one if not found
        db_entry = core_functions.get_chat_mapping("telegram", telegram_chat_id, assistant_id)
        logging.info(f"DB entry: {db_entry}")

        thread_id = core_functions.get_value_from_mapping(db_entry, "thread_id")
        logging.info(f"Thread ID: {thread_id}")

        if not thread_id:
            thread = client.beta.threads.create()
            core_functions.update_chat_mapping("telegram", telegram_chat_id, assistant_id, thread.id)
            thread_id = thread.id
            logging.info(f"New thread created with ID: {thread_id}")

        if not thread_id:
            logging.error("Error: Missing OpenAI thread_id")
            return

        # Construct the prompt for OpenAI with user input and context
        prompt_with_context = (
            f"User Input: {messageEN}\n\n"
            "IMPORTANT: Provide a response under 100 words and format it for a text message, "
            "avoiding backend or technical language."
        )
        logging.info(f"Message for AI: {prompt_with_context} for OpenAI thread ID: {thread_id}")

        # Send the user message to OpenAI
        client.beta.threads.messages.create(thread_id=thread_id, role="user", content=prompt_with_context)
        logging.info("Message sent to OpenAI")

        # Define truncation strategy for conversation context
        truncation_strategy = {"type": "last_messages", "last_messages": 10}

        # Create a run to process the conversation
        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id, truncation_strategy=truncation_strategy)
        logging.info("Run created successfully")

        # Process any action requests from the AI
        core_functions.process_tool_calls(client, thread_id, run.id)
        logging.info("Tool calls processed")

        # Retrieve the latest messages from the AI
        messages = client.beta.threads.messages.list(thread_id=thread_id, limit=10)
        logging.info("Messages retrieved successfully")

        # Extract the AI response from the latest message
        AIresponse = messages.data[0].content[0].text.value
        AIresponse = str(AIresponse)
        logging.info(f"AI Response: {AIresponse}")

        # Save the interaction to Google Sheets
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        append_to_sheet1([telegram_chat_id, messageEN, AIresponse, timestamp])

        # Send the response back to the user on Telegram
        bot.send_message(telegram_chat_id, AIresponse, parse_mode='Markdown')
    
    # Remove any existing webhook before starting polling
    bot.remove_webhook()

    # Start polling in a separate thread
    from threading import Thread
    def start_polling():
        bot.infinity_polling(none_stop=True)
    Thread(target=start_polling).start()
# ====================================================
# Utility Functions
# ====================================================

def convert_ogg_to_wav(ogg_path, wav_path):
    """
    Convert an OGG audio file to WAV format.

    Args:
        ogg_path (str): Path to the OGG file.
        wav_path (str): Path to save the WAV file.
    """
    try:
        data, samplerate = sf.read(ogg_path)
        sf.write(wav_path, data, samplerate)
        logging.info(f"Successfully converted {ogg_path} to {wav_path}")
    except Exception as e:
        logging.error(f"Error converting {ogg_path} to WAV: {e}")

# Retry settings for download functions
retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(requests.exceptions.RequestException)
}

@retry(**retry_settings)
def download_audio(bot, file_path):
    """
    Download an audio file from Telegram.

    Args:
        bot: Telegram bot instance.
        file_path (str): Path to the file.

    Returns:
        bytes: The downloaded audio file content.
    """
    file_info = bot.get_file(file_path)
    return bot.download_file(file_info.file_path)

@retry(**retry_settings)
def download_file(url):
    """
    Download a file from the specified URL.

    Args:
        url (str): The URL of the file.

    Returns:
        bytes: The content of the downloaded file.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.content

