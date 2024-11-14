# TelegramChatbot: AI-Powered Assistant Bot with Google Drive Integration

A versatile Telegram chatbot integrated with OpenAI's ChatGPT, Google Drive API, and WhisperAI for a multi-functional bot. This bot can interpret text, images, and audio messages, storing all interactions in a Google Sheets document for easy data management and analysis. The bot is highly customizable and can be adapted to different use cases.

## 🚀 Features

- **Text Interpretation**: Processes and responds to text messages using OpenAI's ChatGPT API.
- **Image Analysis**: Uses DALL-E API to analyze images sent to the bot and generate detailed responses.
- **Audio Interpretation**: Uses Whisper AI for transcribing and interpreting voice messages.
- **Data Logging**: Logs all user interactions (text, images, audio) into a Google Sheets document using the Google Drive API.
- **Customizable Responses**: Easily change the bot's responses, name, and other parameters.
- - **Basic Flask Web Server**: A small Flask application that currently serves a simple text description, with potential for future enhancements.


## 🛠️ Technologies Used

- **Python**: Main programming language for the bot's logic.
- **Telegram API**: For interacting with users on Telegram.
- **OpenAI ChatGPT API**: Provides natural language understanding and response generation. This API integrates the DALL-E module and the Whisper Feature for transcription.
- **Google Drive API & Google Sheets API**: Logs user interactions in a Google Sheets document.

## 📋 Prerequisites

To use this project, you need the following:

- Python 3.8 or higher
- A Telegram bot token (via BotFather)
- API keys for:
  - OpenAI (for ChatGPT, Whisper and DALL-E) **PAYING**
  - Google Drive and Google Sheets API
- A Google Sheet document set up with the DriveSheets API
- Langchain profile & API(optional)

## ⚙️ Setup Instructions

Follow these steps to set up the Telegram chatbot and connect it with the required APIs:

### 1. Clone the Repository

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/carmenrobres/TelegramChatbot.git
cd TelegramChatbot
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment for managing dependencies. Run the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a .env file in the root directory (where you have main.py) and add your API keys:

```bash
TELEGRAM_API_KEY=your-telegram-api-key
TELEGRAM_CHAT_ID=your-telegram-api-key
OPENAI_API_KEY=your-openai-api-key

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langchain-api-key"
LANGCHAIN_PROJECT="CreAI"
```

***Check up the API Configuration Guide to get your APIs***
Langchain is not really needed but it can be a key element after if you have a big data base.

### 5. Replace Google API file
- Go to the Google Cloud Console.
- Create a new project or select an existing one.
- Enable the Google Drive API and Google Sheets API.
- Download the JSON credentials file and save it as client_key.json in the root directory.
  
### 6. Run the Bot
```bash
python main.py
```

## 🔧 Customization

## 📖 API Configuration Guide

## 📂 Project Structure
```bash
TelegramChatbot/
├── main.py                 # Main script to run the bot
├── sheets_client.py        # Script to append information to Google Drive
├── core_functions.py       # Main bot logic
├── assistant.py            # AI response handling
├── requirements.txt        # Python dependencies
├── client_key.json        # Google API 
├── templates/                 
    └── index.html             # structure for Flask APP
├── resources/                 
    └── DataSet.json           # Data Set of the Assistant Bot **Only works as Json**
├── integrations/                 
    └── Telegram.py             # Telegram message handlers
├── assistant/                 
    └── instructions.txt        # Assistant Instructions
├── docs/static/                 
    └── instructions.txt        # Assistant Instructions
└── .env                    # Environment variables

```

## 🤖 Usage

## How it Works

## References
