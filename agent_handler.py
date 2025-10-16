# agent_handler.py

import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

import config
from chatbot import Chatbot # We use the RAG chatbot directly

def create_solution_agent():
    """
    Creates a specialized agent whose ONLY job is to find solutions.
    It does not manage conversation flow.
    """
    print("Initializing specialized solution-finding agent...")

    rag_chatbot = Chatbot(index_path=config.INDEX_PATH)

    @tool
    def knowledge_base_tool(query: str) -> str:
        """
        Searches the internal knowledge base for solutions to user problems.
        Returns the answer and its sources.
        """
        answer, sources = rag_chatbot.ask(question=query)
        # Combine answer and sources for a complete response
        response = f"{answer}"
        if sources:
            response += f"\n\nSources:{sources}"
        return response

    # Define the tool list
    all_tools = [knowledge_base_tool]
    if os.environ.get("TAVILY_API_KEY"):
        tavily_search = TavilySearchResults(max_results=2, name="internet_search")
        all_tools.append(tavily_search)

    # A much simpler prompt for a specialized task
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at finding solutions. Your job is to answer the user's technical question or problem statement.\n"
            "1. First, you MUST use the `knowledge_base_tool` to search for an answer.\n"
            "2. If the knowledge base does not provide a sufficient answer, you MAY use the `internet_search` tool.\n"
            "3. Your final answer should be the solution you found. Do not add any conversational text like 'Here is the solution' or 'I hope this helps'. Respond only with the solution itself."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    llm = ChatGroq(model=config.CHAT_MODEL, temperature=0.1)
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True)

    print("✅ Solution-finding agent initialized successfully.")
    return agent_executor