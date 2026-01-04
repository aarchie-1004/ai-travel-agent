# db.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env file from the current directory (root)
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    try:
        import streamlit as st
        MONGO_URI = st.secrets["MONGO_URI"]
    except Exception:
         raise ValueError("MONGO_URI not set in .env file or Streamlit secrets.")

try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    print("DEBUG: Successfully connected to MongoDB.")
    db = client['travel_consulting_db'] 
    users_collection = db['users']
    queries_collection = db['queries']
except Exception as e:
    print(f"ERROR: Could not connect to MongoDB: {e}")
    raise