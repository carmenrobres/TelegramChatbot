# sheets_client.py

import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai

# ====================================================
# Configuration
# ====================================================

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Define the scope for Google Sheets and Google Drive API access
scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Specify the path to your service account key file
file_name = 'client_key.json'

# ====================================================
# Google Sheets Initialization
# ====================================================

# Create a ServiceAccountCredentials object using the JSON key file
creds = ServiceAccountCredentials.from_json_keyfile_name(file_name, scope)

# Authorize the gspread client with the credentials
client = gspread.authorize(creds)

# Open the Google Sheet named 'DataThesis'
sheet = client.open('DataThesis')
sheet1 = sheet.get_worksheet(0)  # Access the first worksheet

# ====================================================
# Function Definitions
# ====================================================

def append_to_sheet1(values):
    """
    Appends a list of values as a new row to the first worksheet (Sheet1).
    
    Args:
        values (list): List of values to append as a row.
    """
    try:
        sheet1.append_row(values)
        print("Data appended to Sheet1 successfully.")
    except gspread.exceptions.SpreadsheetNotFound as e:
        print(f"Spreadsheet not found: {e}")
        print("Ensure the spreadsheet name is correct and the service account has access.")
    except gspread.exceptions.APIError as e:
        print(f"An API error occurred: {e}")
        print("Ensure the Google Drive API is enabled and the service account has access to the sheet.")


