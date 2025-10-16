# ticket_services.py

import gspread
from groq import Groq
from datetime import datetime
import uuid
import pandas as pd
import streamlit as st
import config

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

try:
    groq_client = Groq(api_key=config.GROQ_API_KEY)
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    groq_client = None

@st.cache_data(ttl=600)
def sheet_as_dataframe():
    """Fetches all data from the Google Sheet and returns it as a pandas DataFrame."""
    try:
        sheet = authenticate_gspread()
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        print(f"[ERROR] Could not fetch or process Google Sheet data: {e}")
        return pd.DataFrame()

def authenticate_gspread():
    """Connects to Google Sheets using service account credentials."""
    print("Authenticating with Google Sheets...")
    try:
        gc = gspread.service_account(filename=config.GOOGLE_CREDENTIALS_FILE)
        worksheet = gc.open(config.GOOGLE_SHEET_NAME).sheet1
        print("✅ Google Sheets authentication successful.")
        return worksheet
    except Exception as e:
        print(f"[ERROR] Failed to authenticate Google Sheets: {e}")
        raise

def search_previous_tickets_by_email(email: str) -> str:
    """
    Searches the Google Sheet for tickets by email and returns a formatted summary.
    """
    print(f"Searching for previous tickets for email: {email}...")
    df = sheet_as_dataframe()
    if df.empty or 'ticket_by' not in df.columns:
        return "Could not search for previous tickets."

    # Filter for the user's tickets, ignoring case
    user_tickets = df[df['ticket_by'].str.lower() == email.lower()]

    if user_tickets.empty:
        return "No previous tickets found for this user."

    # Format the results into a concise string for the agent's context
    summary = "Here is a summary of the user's previous tickets:\n"
    for _, row in user_tickets.head(5).iterrows(): # Limit to last 5 for brevity
        summary += (
            f"- Ticket ID: {row.get('ticket_id', 'N/A')[:8]}, "
            f"Status: {row.get('ticket_status', 'N/A')}, "
            f"Issue: '{row.get('ticket_content', 'N/A')}'\n"
        )
    return summary

def create_ticket(sheet, content: str, email: str):
    """Appends a new ticket row to the Google Sheet and returns its ID."""
    print(f"Creating ticket for {email}...")
    ticket_id = str(uuid.uuid4())
    category = classify_ticket_content(content)
    new_row = [
        ticket_id,
        content,
        category,
        datetime.now().isoformat(),
        email,
        "In Progress",
        "", "", "", ""
    ]
    sheet.append_row(new_row)
    print(f"✅ Ticket {ticket_id} created.")
    return ticket_id

def update_ticket_solution(sheet, ticket_id: str, headers: list, solution: str, sources: str):
    """Finds a ticket by ID and updates it with the proposed solution."""
    print(f"Updating ticket {ticket_id} with solution...")
    try:
        cell = sheet.find(ticket_id)
        if cell:
            solution_col = headers.index("solution") + 1
            solution_with_sources = f"{solution}\n\n{sources}" if sources else solution
            sheet.update_cell(cell.row, solution_col, solution_with_sources)
            print(f"✅ Ticket {ticket_id} updated with solution.")
    except Exception as e:
        print(f"[ERROR] Could not update ticket {ticket_id} with solution: {e}")

# Restored from the old project for simple status changes
def update_ticket_status(sheet, ticket_id: str, headers: list, status: str):
    """Finds a ticket by ID and updates only its status."""
    print(f"Updating status for ticket {ticket_id} to '{status}'...")
    try:
        cell = sheet.find(ticket_id)
        if cell:
            status_col = headers.index("ticket_status") + 1
            sheet.update_cell(cell.row, status_col, status)
            print(f"✅ Ticket {ticket_id} status set to '{status}'.")
    except Exception as e:
        print(f"[ERROR] Could not update ticket {ticket_id} status: {e}")

def update_ticket_feedback(sheet, ticket_id: str, headers: list, status: str, feedback: str, conversation_history: str):
    """Updates the ticket status, feedback, and sentiment."""
    print(f"Updating feedback for ticket {ticket_id}...")
    try:
        cell = sheet.find(ticket_id)
        if not cell:
            print(f"[ERROR] Ticket ID {ticket_id} not found.")
            return

        sentiment = analyze_conversation_sentiment(conversation_history)
        sheet.update_cell(cell.row, headers.index("ticket_status") + 1, status)
        sheet.update_cell(cell.row, headers.index("Customer_Feedback") + 1, feedback)
        sheet.update_cell(cell.row, headers.index("Feedback_Timestamp") + 1, datetime.now().isoformat())
        sheet.update_cell(cell.row, headers.index("Sentiment") + 1, sentiment)
        print(f"✅ Ticket {ticket_id} finalized with status '{status}'.")
    except Exception as e:
        print(f"[ERROR] Could not update ticket {ticket_id} feedback: {e}")

def classify_ticket_content(content: str):
    """Uses a Groq model to classify the ticket content."""
    if not groq_client: return "Unclassified"
    print(f"Classifying: '{content[:40]}...'")
    categories = ["maintainance", "product support", "refund", "high_priority_product"]
    system_prompt = f"Categorize the content into one of: {', '.join(categories)}. Respond with only the category name."
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ], model=config.CLASSIFICATION_MODEL, temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Could not classify content: {e}")
        return "Unclassified"

def analyze_conversation_sentiment(conversation: str):
    """Analyzes conversation history to determine user sentiment."""
    if not groq_client: return "neutral"
    print("Analyzing sentiment...")
    system_prompt = "Analyze the user's final sentiment from the conversation. Respond with only: 'positive', 'negative', or 'neutral'."
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation}
            ], model=config.CLASSIFICATION_MODEL, temperature=0.1
        )
        sentiment = completion.choices[0].message.content.strip().lower()
        return sentiment if sentiment in ["positive", "negative", "neutral"] else "neutral"
    except Exception as e:
        print(f"[ERROR] Could not analyze sentiment: {e}")
        return "neutral"
    
def send_slack_notification(ticket_id: str, content: str, email: str):
    """Sends a notification to a Slack channel when a ticket is escalated."""
    if not config.SLACK_BOT_TOKEN:
        print("[WARN] SLACK_BOT_TOKEN not set. Skipping notification.")
        return

    client = WebClient(token=config.SLACK_BOT_TOKEN)
    message = (
        f"🚨 *Ticket Escalation Alert* 🚨\n\n"
        f"*Ticket ID:* `{ticket_id}`\n"
        f"*User Email:* {email}\n"
        f"*Problem:* {content}\n\n"
        f"A human agent needs to follow up on this ticket."
    )

    try:
        print(f"Sending Slack notification for ticket {ticket_id}...")
        client.chat_postMessage(channel=config.SLACK_CHANNEL_ID, text=message)
        print("✅ Slack notification sent successfully.")
    except SlackApiError as e:
        print(f"[ERROR] Failed to send Slack notification: {e.response['error']}")