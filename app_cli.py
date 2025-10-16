# app_cli.py

import os
import gspread
import uuid

import config
from document_processor import find_files, load_documents, split_documents, build_and_save_faiss
from ticket_services import authenticate_gspread
from agent_handler import create_agent_executor # Import the new agent

def run_cli_app():
    # --- One-time setup: Indexing ---
    if config.REBUILD_INDEX or not os.path.exists(config.INDEX_PATH):
        print("--- Building Index ---")
        doc_paths = find_files(config.DOCS_PATH)
        if doc_paths:
            documents = load_documents(doc_paths)
            chunks = split_documents(documents)
            if chunks:
                build_and_save_faiss(chunks)
        else:
            print("No documents found in 'my_docs' to index.")
    
    if not config.INDEX_PATH.exists():
        raise FileNotFoundError("Index not found. Please run 'python document_processor.py' or set REBUILD_INDEX = True.")
        
    # --- Initialize services ---
    # Agent now handles interactions, but we might need sheet for other direct operations later
    sheet = authenticate_gspread() 
    agent_executor = create_agent_executor()
    
    print(f"\n--- Conversational Agent is running (type 'exit' to quit) ---")
    
    # The agent now manages its own state and memory. We just need a session_id.
    session_id = str(uuid.uuid4())
    chat_history = []

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            if not user_input:
                continue

            # Invoke the agent with the user input and chat history
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            answer = response.get("output", "Sorry, I encountered an error.")
            
            # Add interaction to history for the agent's context
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})

            print(f"\nBot: {answer}\n")

    except KeyboardInterrupt:
        print("\n--- Agent stopped by user. ---")
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"[ERROR] Spreadsheet '{config.GOOGLE_SHEET_NAME}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    run_cli_app()