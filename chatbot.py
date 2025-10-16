from pathlib import Path
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Import constants from our single config file
import config

# --- HELPER FUNCTION TO FORMAT SOURCES (from Code 1) ---
def format_sources(docs: list, markdown_format: bool = False):
    """Formats the source documents for display."""
    if not docs:
        return ""
    
    lines = []
    for doc in docs:
        source_path = Path(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        
        if markdown_format:
            line = f"- **Source:** {source_path.name}"
            if page is not None:
                line += f" (Page: {page + 1})"
        else:
            line = f"- Source: {source_path.name}"
            if page is not None:
                line += f" (Page: {page + 1})"
        lines.append(line)
        
    return "\n" + "\n".join(lines)


# --- THE FULLY FEATURED CHATBOT CLASS ---
class Chatbot:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.vector_store = self._load_vector_store()
        # This dictionary will store simple chat histories for each session
        self.chat_histories = {}
        self.chain = self._create_rag_chain()

    def _load_vector_store(self) -> FAISS:
        """Loads the FAISS index."""
        if not self.index_path.exists():
            # Restored helpful error message from Code 1
            raise FileNotFoundError(
                f"Index path not found: {self.index_path.resolve()}\n"
                f"Please run 'python document_processor.py' first to create the index."
            )
        
        print(f"Loading existing index from {self.index_path.resolve()}...")
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        vector_store = FAISS.load_local(
            str(self.index_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ Index loaded successfully.")
        return vector_store

    def _create_rag_chain(self):
        """Creates the RAG chain using modern LCEL and a custom prompt."""
        print("Creating the RAG chain...")
        
        chat_model = ChatGroq(model=config.CHAT_MODEL, temperature=0.2)
        
        # Restored detailed prompt from Code 1
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a concise, careful assistant. Answer ONLY from the provided context. If the answer is not in the context, say you don't know. Cite sources by filename and page if present."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question:\n{input}\n\nContext:\n{context}"),
        ])
        
        doc_chain = create_stuff_documents_chain(chat_model, prompt)
        retriever = self.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        def get_session_history(session_id: str):
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = ChatMessageHistory()
            return self.chat_histories[session_id]

        conversational_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        print("✅ RAG chain created successfully.")
        return conversational_chain

    def ask(self, question: str, session_id: str = "default_session", markdown_sources: bool = False):
        """Asks a question to the chatbot and returns the answer and sources."""
        if not question:
            return "Please enter a question.", ""
        
        response = self.chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        sources = format_sources(response.get("context", []), markdown_format=markdown_sources)
        return answer, sources

    # New feature from Code 2 is kept
    def get_history_as_string(self, session_id: str) -> str:
        """Retrieves and formats the chat history for a given session."""
        history = self.chat_histories.get(session_id)
        if not history or not history.messages:
            return "No history found for this session."
        
        formatted_history = []
        for msg in history.messages:
            role = "User" if msg.type == "human" else "Assistant"
            formatted_history.append(f"{role}: {msg.content}")
        return "\n".join(formatted_history)

# --- MAIN EXECUTION BLOCK FOR TESTING (from Code 1) ---
if __name__ == '__main__':
    print("--- Chatbot Backend Initializing (Test Mode) ---")
    
    # Assuming config.py has INDEX_PATH defined, e.g., INDEX_PATH = Path("faiss_index")
    if not config.INDEX_PATH.exists():
        print("No index found. Please run 'python document_processor.py' first.")
    else:
        chatbot = Chatbot(index_path=config.INDEX_PATH)

        print("\n--- Starting Conversation (type 'exit' to quit) ---")
        session_id_test = "cli_test" # Define a session_id for testing
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            answer, sources = chatbot.ask(user_input, session_id=session_id_test)
            
            print(f"\nBot: {answer}")
            if sources:
                print(f"Sources:{sources}")
            print("\n" + "="*50)