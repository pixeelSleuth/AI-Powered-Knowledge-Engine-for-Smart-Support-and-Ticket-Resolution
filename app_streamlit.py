# app_streamlit.py

import streamlit as st
import pandas as pd
import re
from collections import Counter
import uuid

from ticket_services import send_slack_notification

import config
from ticket_services import (
    authenticate_gspread,
    create_ticket,
    update_ticket_solution,
    update_ticket_status,
    update_ticket_feedback,
    classify_ticket_content,
    sheet_as_dataframe
)
from agent_handler import create_solution_agent # Import the new simplified agent
from chatbot import format_sources
from ticket_services import search_previous_tickets_by_email

# --- Constants for Conversation State ---
ASK_FOR_PROBLEM = "ASK_FOR_PROBLEM"
ASK_FOR_EMAIL = "ASK_FOR_EMAIL"
AWAITING_FOLLOW_UP = "AWAITING_FOLLOW_UP"
CONVERSATION_DONE = "CONVERSATION_DONE"

# =======================================================================
# INITIALIZE RESOURCES (Cached for performance)
# =======================================================================

@st.cache_resource
def initialize_resources():
    """Loads services and initializes the LangChain agent."""
    sheet = authenticate_gspread()
    try:
        headers = sheet.row_values(1)
    except Exception as e:
        st.error(f"Could not load headers from Google Sheet: {e}")
        headers = [] # Default to empty list on failure
    solution_agent = create_solution_agent()
    return sheet, headers, solution_agent

# Initialize everything
sheet, headers, solution_agent = initialize_resources()


# =======================================================================
# UI COMPONENT: THE CHATBOT INTERFACE (Now with a State Machine)
# =======================================================================
def chatbot_interface():
    st.subheader("🤖 AI Support Agent")

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Please describe your technical problem."}]
    if "conversation_stage" not in st.session_state:
        st.session_state.conversation_stage = ASK_FOR_PROBLEM
    if "ticket_id" not in st.session_state:
        st.session_state.ticket_id = None
    if "user_problem" not in st.session_state:
        st.session_state.user_problem = ""

    # --- Display existing messages ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle user input based on conversation stage ---
    if prompt := st.chat_input("Your response..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # === STATE MACHINE LOGIC ===

        # Stage 1: User has provided the problem description
        if st.session_state.conversation_stage == ASK_FOR_PROBLEM:
            st.session_state.user_problem = prompt
            st.session_state.conversation_stage = ASK_FOR_EMAIL
            response = "Thank you. To create a support ticket, please provide your email address."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Stage 2: User has provided their email
        elif st.session_state.conversation_stage == ASK_FOR_EMAIL:
            email = prompt
            with st.spinner("Creating ticket and finding a solution..."):
                ticket_id = create_ticket(sheet, st.session_state.user_problem, email)
                st.session_state.ticket_id = ticket_id

                # B) NEW: Search for the user's ticket history
                ticket_history = search_previous_tickets_by_email(email)

                # C) NEW: Build a smarter prompt for the agent
                agent_prompt = (
                    f"A user has this problem: '{st.session_state.user_problem}'\n\n"
                    "Please provide a solution. "
                    f"For additional context, here is their previous ticket history:\n{ticket_history}"
                )
                # D) Call the agent with the enhanced prompt
                agent_response = solution_agent.invoke({"input": agent_prompt})
                solution = agent_response.get("output", "I could not find a specific solution in the knowledge base.")

                update_ticket_solution(sheet, ticket_id, headers, solution, "")

            response = (f"{solution}\n\n---\nTicket **#{ticket_id[:8]}** has been created. "
                        f"Did this solution resolve your issue? Please reply with **'done'** or **'escalate'**.")
            st.session_state.conversation_stage = AWAITING_FOLLOW_UP
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Stage 3: Awaiting feedback or follow-up questions
        elif st.session_state.conversation_stage == AWAITING_FOLLOW_UP:
            # This is where the fix is: Define user_response right away
            user_response = prompt.lower().strip()

            # Check for "done"
            if user_response in ["done", "yes", "resolved", "thanks"]:
                with st.spinner("Finalizing ticket..."):
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    update_ticket_feedback(sheet, st.session_state.ticket_id, headers, "Resolved", prompt, history_str)
                response = "Great! I've marked the ticket as resolved. Have a great day!"
                st.session_state.conversation_stage = CONVERSATION_DONE
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

            # Check for "escalate"
            
            elif user_response in ["agent", "escalate", "human", "escalate to agent"]:
                with st.spinner("Escalating ticket..."):
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    
                    # Update the ticket in Google Sheets
                    update_ticket_feedback(sheet, st.session_state.ticket_id, headers, "Escalated", prompt, history_str)
                    
                    # NEW: Send the Slack notification
                    send_slack_notification(
                        ticket_id=st.session_state.ticket_id,
                        content=st.session_state.user_problem,
                        email=st.session_state.messages[3]['content'] # A bit fragile, assumes email is always the 4th message
                    )

                response = "Understood. I have escalated your ticket to a human agent. They will contact you via email shortly. Thank you for your patience."
                st.session_state.conversation_stage = CONVERSATION_DONE
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

            # If it's not "done" or "escalate", treat it as a follow-up
            else:
                 with st.spinner("Finding an answer to your follow-up..."):
                    agent_response = solution_agent.invoke({"input": prompt})
                    solution = agent_response.get("output", "I'm not sure how to answer that.")
                 response = (f"{solution}\n\n---\n"
                             f"Does this help? Please reply with **'done'** or **'escalate'** if you are finished.")
                 st.session_state.messages.append({"role": "assistant", "content": response})
                 st.rerun()

# =======================================================================
# UI COMPONENT: ANALYTICAL DASHBOARD
# =======================================================================
# =======================================================================
# UI COMPONENT: ANALYTICAL DASHBOARD (UPGRADED)
# =======================================================================
def analytical_dashboard():
    st.subheader("📊 Analytical Dashboard")
    st.write("This dashboard provides insights into support tickets and identifies potential knowledge gaps.")

    df = sheet_as_dataframe()

    if df.empty:
        st.warning("Could not load data from Google Sheets, or the sheet is empty.")
        return

    # --- NEW: Interactive Filters in the Sidebar ---
    st.sidebar.header("Dashboard Filters")
    
    # Ensure 'ticket_category' column exists for the filter
    if 'ticket_category' in df.columns:
        # Get unique categories from the dataframe, handling potential blank values
        unique_categories = df['ticket_category'].dropna().unique()
        selected_categories = st.sidebar.multiselect(
            "Filter by Ticket Category:",
            options=unique_categories,
            default=unique_categories  # Default to all categories selected
        )
        # Filter the dataframe based on selection
        df = df[df['ticket_category'].isin(selected_categories)]
    else:
        st.sidebar.warning("'ticket_category' column not found.")
    
    # --- Convert timestamp column for time-series analysis ---
    if 'ticket_timestamp' in df.columns:
        df['ticket_timestamp'] = pd.to_datetime(df['ticket_timestamp'])
    else:
        st.error("The Google Sheet is missing the 'ticket_timestamp' column, which is required for charts.")
        return

    # --- End of Sidebar ---
    
    # --- NEW: Ticket Volume Over Time Chart ---
    st.markdown("---")
    st.markdown("### 📈 Ticket Volume Over Time")
    st.write("Shows the number of new tickets created each day to help identify trends.")
    tickets_over_time = df.set_index('ticket_timestamp').resample('D').size().rename("Number of Tickets")
    st.line_chart(tickets_over_time)

    # --- Existing Low-Coverage Alerts (Now Filterable) ---
    st.markdown("---")
    st.markdown("### 🚨 Low-Coverage Alerts")
    st.write("Tickets that were escalated or received negative feedback.")
    
    required_alert_cols = ['ticket_status', 'Sentiment']
    if not all(col in df.columns for col in required_alert_cols):
        st.error("Sheet is missing 'ticket_status' and/or 'Sentiment' columns.")
        return

    low_coverage_df = df[
        (df['ticket_status'] == 'Escalated') |
        (df['Sentiment'] == 'negative')
    ]

    if low_coverage_df.empty:
        st.success("No low-coverage areas detected for the selected filters!")
    else:
        st.metric(label="Alerts Requiring Attention", value=len(low_coverage_df))
        display_cols = ['ticket_id', 'ticket_content', 'ticket_status', 'Sentiment', 'ticket_by']
        st.dataframe(low_coverage_df[[col for col in display_cols if col in df.columns]])

    # --- NEW: Escalations by Category Chart ---
    st.markdown("---")
    st.markdown("### Category Performance")
    st.write("Shows which ticket categories are being escalated most often.")
    
    if 'ticket_category' in df.columns:
        escalated_df = df[df['ticket_status'] == 'Escalated']
        if escalated_df.empty:
            st.info("No tickets have been escalated for the selected filters.")
        else:
            escalations_by_category = escalated_df['ticket_category'].value_counts()
            st.bar_chart(escalations_by_category)
    
    # --- Existing Referenced Articles (Now Filterable) ---
    st.markdown("---")
    st.markdown("### 📚 Most Frequently Referenced Knowledge Articles")
    st.write("Shows which documents are most often cited in successful solutions.")
    
    if 'solution' not in df.columns:
        st.error("Sheet is missing the 'solution' column.")
        return
        
    source_pattern = r"Sources:\s*([\w\d\._-]+)"
    all_sources = df['solution'].str.findall(source_pattern).explode().dropna()

    if all_sources.empty:
        st.info("No sources have been referenced in tickets for the selected filters.")
    else:
        source_counts = Counter(all_sources)
        source_df = pd.DataFrame(source_counts.items(), columns=['Article', 'Count']).sort_values('Count', ascending=False)
        st.bar_chart(source_df.set_index('Article'))

# =======================================================================
# UI COMPONENT: VALIDATION TOOLS
# =======================================================================
def validation_tools():
    # This function remains unchanged and should work correctly
    st.subheader("🔎 Live Tagging Validation")
    st.write("Test the LLM's classification logic in real-time.")
    sample_text = st.text_area("Enter sample ticket content:", "My screen is cracked.", height=150)
    if st.button("Classify Content"):
        if not sample_text:
            st.warning("Please enter some content to classify.")
        else:
            with st.spinner("🧠 Analyzing content..."):
                category = classify_ticket_content(sample_text)
                st.success(f"**Predicted Category:** `{category}`")


# =======================================================================
# MAIN APP LAYOUT
# =======================================================================
def main():
    st.set_page_config(page_title="AI Support System", page_icon="⚙️", layout="wide")
    st.title("⚙️ AI-Powered Support System")

    tab1, tab2, tab3 = st.tabs(["💬 Chat Agent", "📊 Analytical Dashboard", "🔎 Validation Tools"])

    with tab1:
        chatbot_interface()
    with tab2:
        analytical_dashboard()
    with tab3:
        validation_tools()

if __name__ == '__main__':
    main()