import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# --- API Keys ---

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

# Check for essential keys
if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY is not set in your .env file.")

# --- File Paths ---
DOCS_PATH = Path("./my_docs")
INDEX_PATH = Path("./faiss_index")
GOOGLE_CREDENTIALS_FILE = "credentials.json"

# --- Model Configuration ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.1-8b-instant"
CLASSIFICATION_MODEL = "llama-3.1-8b-instant"

# --- Document Processing Config ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Google Sheets Config ---
GOOGLE_SHEET_NAME = "AITicketSystem"

SLACK_CHANNEL_ID = "" # Paste your Channel ID here

# --- Application Flags ---
REBUILD_INDEX = False