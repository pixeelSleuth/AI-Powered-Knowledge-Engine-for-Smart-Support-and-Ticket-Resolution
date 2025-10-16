# 🧠 AI-Powered Knowledge Engine for Smart Support & Ticket Resolution

## ℹ️ Project Overview

This project implements an intelligent, AI-based **Knowledge Management Platform** designed to enhance the efficiency and consistency of customer support operations. By leveraging **Large Language Models (LLMs)** from **Groq** and a **Retrieval-Augmented Generation (RAG)** architecture, the system automatically processes and analyzes support ticket content to provide **real-time, context-aware article recommendations** and proactively identify knowledge base deficiencies.

The platform's core goal is to **improve first-response quality, reduce average resolution time (ART), and ensure the knowledge base is always aligned with current customer needs.**

## ✨ Key Features & Achieved Outcomes

| Feature Area | Description | Status |
| :--- | :--- | :--- |
| **Knowledge Categorization & Tagging** | Uses **Groq's Llama-3.1-8b-instant** model for automated, semantic classification of support articles. | ✅ Complete (Milestone 2) |
| **Real-Time Recommendation System** | Implemented via a specialized **LangChain Agent** that searches the **FAISS Vector Store (RAG)** to instantly suggest relevant articles. | ✅ Complete (Milestone 3) |
| **Content Gap Detection** | Logic implemented in Streamlit dashboard to flag escalated tickets and negative sentiment as low-coverage areas. | ✅ Complete (Milestone 3) |
| **Data & Reporting Hub** | **Streamlit** dashboard visualizes ticket volume, category performance, and source usage, with **Google Sheets** as the primary data backend. | ✅ Complete (Milestone 4) |

---

## 🛠 Technology Stack

This project is built primarily on Python and its data/AI ecosystem:

* **Core Language:** Python 3.10+
* **LLMs & Orchestration:** **Groq** (for fast inference), **LangChain** (for RAG and Agent logic).
* **Vector Store & Embeddings:** **FAISS** (vector database) and **HuggingFace Embeddings**.
* **Data & Persistence:** **Google Sheets API** (`gspread`), **Pandas**.
* **Frontend / Dashboard:** **Streamlit** (`app_streamlit.py`).
* **Tools:** Git, VS Code.

---

## 🚀 Getting Started (Run Locally)

These instructions detail how to set up and run the Streamlit application for the AI Support Agent.

### Prerequisites

* Python 3.10+
* A **Groq API Key** for LLM inference (Classification and Agent).
* A **Google Sheets** Service Account `credentials.json` file for data read/write.
* Your knowledge documents must be placed in a local folder named **`my_docs`**.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL - To be inserted after upload]
    cd ai-knowledge-engine
    ```

2.  **Set up the Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing all packages like `streamlit`, `langchain`, `groq`, `gspread`, `pandas`, etc., before running this step.)*

3.  **Configure Environment:**
    * Place your **Google Sheets Service Account JSON file** (`credentials.json`) in the root directory.
    * Create a file named **`.env`** in the root directory and add your API keys and configuration settings as defined in `config.py`:
        ```dotenv
        GROQ_API_KEY="your-groq-api-key"
        # Optional: TAVILY_API_KEY="your-tavily-key" for external search tool
        # Optional: SLACK_BOT_TOKEN="xoxb-..." 
        ```

4.  **Build the Knowledge Index:**
    You must build the vector store index before running the app.
    ```bash
    python document_processor.py
    ```

### Running the Application

To start the Streamlit web application:

```bash
streamlit run app_streamlit.py