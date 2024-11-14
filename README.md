# TelegramChatbot: AI-Powered Assistant Bot with Google Drive Integration

A versatile Telegram chatbot integrated with OpenAI's ChatGPT, Google Drive API, and WhisperAI for a multi-functional bot. This bot can interpret text, images, and audio messages, storing all interactions in a Google Sheets document for easy data management and analysis. The bot is highly customizable and can be adapted to different use cases. With a personalized DataBase you can prompt any type of bot to answer as your assistant bot. 

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

*Check up the API Configuration Guide to get your APIs*.

Langchain is not really needed but it can be a key element after if you have a big data base.

### 5. Replace Google API file
*Check up the API Configuration Guide to get more information*
- Go to the [Google Cloud Console.](https://console.cloud.google.com/)
- Create a new project or select an existing one.
- Enable the Google Drive API and Google Sheets API.
- Download the JSON credentials file and save it as client_key.json in the root directory.
  
### 6. Run the Bot
```bash
python main.py
```

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
├── docs/static/                 # DO NOT TOUCH
├── .storage                 # DO NOT TOUCH - Location where the vector store is saved.
    └── assitant.json          #your chatgpt assitant information. if you earase the file it will create another bot. 
    └── chat_mappings.db
└── .env                    # Environment variables
```

## 🔧 Customization
You can easily customize various aspects of the bot's responses and behavior:

1. **Assistant Role Customization**:
   - In the `Assistant` folder, you’ll find `instructions.txt`. Here, you can define and customize the role of your assistant, tailoring its personality and response style.

2. **Changing the Assistant's Name**:
   - Open `assistant.py` and update the assistant's name on **line 20**. This name will be reflected in your ChatGPT Playground Project.

3. **Telegram Bot Customization**:
   - In `integrations/Telegram.py`, you can modify the following:
     - The **trigger greeting** (line 122): This automated response prevents unnecessary token usage by ChatGPT.
     - The **welcome message** for the `/start` command (line 142).
     - The **welcome message** when a greeting word is detected (line 152).

4. **Database Customization**:
   - The bot uses a JSON file for data storage located at `resources/DataSet.json`. You can modify this file using standard JSON format to update or add new data.

5. **Image Processing Customization**:
   - If you want to change how images are processed, you can update the logic in `integrations/Telegram.py` on **line 250**.

6. **Bot Profile Customization**:
   - To modify the bot’s name, user image, and description, use **@BotFather** on Telegram.

7. **Flask App Customization**:
   - To modify the  information set on the Flask app go to `templates/index.html`. Right now it has a basic text but you can enhance the app by adding more information.


## 📖 API Configuration Guide

### Telegram Bot API
Open Telegram and search for BotFather.

Use the command /newbot and follow the instructions to create your bot.

Copy the token and paste it into your .env file as TELEGRAM_API_KEY.

### OpenAI API
Go to OpenAI's API [website](https://platform.openai.com/docs/overview).

Sign up or log in.

Select a project, go to Dashboard, API keys.

Create a new API key.

Add the API key to your .env file as OPENAI_API_KEY.

### Google Drive API
Create a Google Sheet file. It has to have these columns in this order: `USER |	MESSAGE	 |	ANSWER	 |	TIME`

Go to Google Cloud Console and log in.

Create a new project and enable the Google Drive API and Google Sheets API.

Through API & Services, Create credentials and download the JSON file, saving it as credentials.json.

Share your Google Sheet with the service account email in the JSON file.

Paste the json file as client_key.json in the main folder.


## 🤖 Usage

Once you have the bot set up and running, you can interact with it directly on Telegram. Here’s what you can do:

1. **Text Interaction**:
   - Simply type your message to the bot, and it will respond based on the assistant role defined in `instructions.txt`. The bot personalizes its responses using your Telegram username for a more engaging experience.

2. **Audio Interpretation**:
   - You can send voice messages to the bot. It uses Whisper AI to transcribe the audio and then responds accordingly based on the assistant’s role and personality settings.

3. **Image Interpretation**:
   - Share an image with the bot, and it will analyze it using DALL-E API, providing a detailed response based on the image and text you add.

4. **Automated Greeting**:
   - When you start the chat with the `/start` command or send a greeting, the bot automatically sends a welcome message without using extra tokens from the ChatGPT API.

5.**Basic Flask Server**:
   - The project also includes a small Flask server that currently serves a simple text information. This sets the stage for potential future web-based enhancements.


Enjoy your personalized chatbot experience and feel free to customize the bot’s role to fit your needs!


## References
This project was inspired by and incorporates ideas and structures from similar AI chatbot projects. For additional insights and inspiration, check out the following repositories:

- [WattWise-AI-Bot (Replit Integration)](https://github.com/TheWattWiseProject/WattWise-AI-Bot_REPLIT/tree/main): WattWise is your energy-awareness companion, blending AI technology and personalized advice to help you visualize your daily energy usage across food, transport, and the home. By combining available data and your input on daily habits, Wattwise provides real-time insights and advice to make small changes with a big collective impact.

- [Slack-Example Bot](https://github.com/LAIA-GitHub/Slack-Example): A well-documented example of integrating an AI assistant with Slack, utilizing OpenAI's API and demonstrating effective bot interaction and automation.

These projects served as valuable references in designing the architecture, features, and integrations for this Telegram chatbot.