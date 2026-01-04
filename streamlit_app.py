# streamlit_app.py (in root - simpler UI, Travel theme, with streaming)
import streamlit as st
import time
import datetime
import hashlib
import os
import uuid
import logging

# Import directly from sibling files
from db import users_collection, queries_collection
# --- IMPORT FROM travel_logic.py ---
from travel_logic import get_runnable_graph, GraphState # Import state type too

# Configure logger
logger = logging.getLogger(__name__)


# --- Basic Setup ---
from dotenv import load_dotenv
load_dotenv() # Load .env

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Travel Planner", # Updated Title
    page_icon="✈️",             # Updated Icon
    layout="centered"
)

# Get the compiled graph instance
try:
    graph = get_runnable_graph()
    logger.info("Streamlit app obtained runnable graph.")
except Exception as e:
    st.error(f"FATAL: Could not compile or get the LangGraph: {e}")
    logger.critical(f"Failed to get runnable graph: {e}", exc_info=True)
    st.stop()


# --- Session State ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = None
if 'messages' not in st.session_state: st.session_state.messages = []

# --- Auth Functions --- (Copied from your provided code, assumed correct)
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str):
    if users_collection is None:
        logger.error("users_collection is None (MongoDB connection failed).")
        return False, "Database connection error."
    if users_collection.find_one({"username": username}):
        return False, "❌ Username already exists!"
    hashed_pw = hash_password(password)
    try:
        users_collection.insert_one({"username": username, "password": hashed_pw})
        return True, "✅ Registered successfully!"
    except Exception as e:
        logger.error(f"Failed to register user {username}: {e}", exc_info=True)
        return False, "Database error during registration."


def login_user(username: str, password: str):
    if users_collection is None:
        logger.error("users_collection is None (MongoDB connection failed).")
        return False, "Database connection error."
    try:
        user = users_collection.find_one({"username": username})
        if user and user['password'] == hash_password(password):
            return True, "✅ Login successful!"
        return False, "❌ Invalid username or password!"
    except Exception as e:
        logger.error(f"Failed to login user {username}: {e}", exc_info=True)
        return False, "Database error during login."

# --- Function to Run LangGraph (Returns full result string) ---
# --- RENAME function and update state key ---
def run_travel_graph(user_input: str) -> str:
    global graph
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    logger.info(f"Invoking graph for session {session_id} with input: {user_input}")
    final_state: GraphState = None
    try:
        initial_state = {"original_input": user_input}
        try:
            final_state = graph.invoke(initial_state, config=config)
        except Exception as graph_err:
             logger.error(f"LangGraph invocation failed: {graph_err}", exc_info=True)
             return f"Sorry, an error occurred during graph processing: {graph_err}"

        logger.info(f"Graph invocation finished. Final state: {final_state}")

        # --- Use travel_suggestions key ---
        result_key = "travel_suggestions"
        if final_state is None or result_key not in final_state or final_state.get("error"):
            error_msg = final_state.get("error", "Graph did not produce travel suggestions.") if final_state else "Graph execution failed."
            logger.error(f"Graph failed or produced error: {final_state}")
            return f"Sorry, an error occurred: {error_msg}"

        result_markdown = final_state[result_key]

        # Save to DB (update keys)
        if queries_collection is not None:
            try:
                queries_collection.insert_one({
                    "username": st.session_state.username,
                    "user_input": user_input,
                    "cleaned_criteria": final_state.get("cleaned_criteria"), # Updated key
                    "search_results": final_state.get("search_results"),
                    "ai_response": result_markdown,
                    "timestamp": datetime.datetime.now(),
                    "session_id": session_id
                })
                logger.info("Saved graph results to DB.")
            except Exception as db_e:
                logger.error(f"Failed to save query to DB: {db_e}", exc_info=True)
        else:
            logger.error("queries_collection is None. Cannot save results.")

        # Return only the markdown string for streaming
        return result_markdown

    except Exception as e:
        logger.error(f"Unexpected error in run_travel_graph: {e}", exc_info=True)
        error_context = f" (Partial state: {final_state})" if final_state else ""
        return f"Sorry, a critical error occurred.{error_context}"

# --- Generator for Streaming Effect ---
def stream_response(text: str):
    """Yields words from the text with a 0.05 second delay."""
    if not isinstance(text, str):
        text = str(text)
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.05) # Adjust delay here

# --- Streamlit UI ---

# LOGIN / REGISTER PAGE
if not st.session_state.logged_in:
    # --- Update Title ---
    st.title("✈️ AI Travel Planner - Login / Register")
    choice = st.radio("Select Action", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        if not username or not password: st.error("⚠️ Enter username/password")
        else:
            success, msg = register_user(username, password) if choice == "Register" else login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.messages = []
                st.rerun()
            else: st.error(msg)

# MAIN APP PAGE
if st.session_state.logged_in:
    # --- Update Title ---
    st.title(f"✈️ Welcome, {st.session_state.username}!")
    st.write("Ask for travel ideas (e.g., destination, interests, budget):") # Updated prompt

    # Display existing chat messages from session state
    for message in st.session_state.messages:
        # --- Update Avatars ---
        avatar = "🧑‍✈️" if message["role"] == "user" else "🗺️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input
    # --- Update Placeholder ---
    if user_input := st.chat_input("e.g., Weekend trip near Ludhiana, budget beach vacation..."):
        # Append and display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        # --- Update Avatar ---
        with st.chat_message("user", avatar="🧑‍✈️"):
            st.markdown(user_input)

        # Get the full response from the graph (could include errors)
        # Add a placeholder while the graph runs
        # --- Update Avatar ---
        with st.chat_message("assistant", avatar="🗺️"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🤔 Planning your trip...") # Updated thinking message
            try:
                # --- Call RENAMED function ---
                full_response_text = run_travel_graph(user_input)
                thinking_placeholder.empty() # Remove the thinking message

                # Stream the response using the generator and st.write_stream
                stream_gen = stream_response(full_response_text)
                streamed_content = st.write_stream(stream_gen)

                # Store the complete message in session state *after* streaming
                st.session_state.messages.append({"role": "assistant", "content": streamed_content})

            except Exception as e:
                # Catch any unexpected errors during graph call or streaming
                logger.error(f"Error during graph call or streaming in UI: {e}", exc_info=True)
                thinking_placeholder.error(f"An unexpected error occurred: {e}")
                error_msg_for_chat = f"Sorry, a critical error occurred during processing: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg_for_chat})

        # No automatic rerun needed here, write_stream updates the placeholder