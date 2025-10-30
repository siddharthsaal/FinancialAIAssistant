# agents.py

import os
import re
import requests
import traceback
import pandas as pd
from typing import TypedDict, List, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

from vanna_util import get_vanna_instance

# --- Configuration and Models ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "") # Use environment variables
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Initialize the LLM model using LangChain for consistency
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_API_URL)
vn = get_vanna_instance()

# --- Agent State Definition ---
# This class defines the "memory" of our agentic system as it processes a query.
class AgentState(TypedDict):
    original_question: str
    processed_question: str
    is_arabic: bool
    classification: str
    chat_history: List[BaseMessage]
    sql_query: str
    df_results: pd.DataFrame
    final_answer: str

# --- Agent Nodes (Functions that perform actions) ---

def router_node(state: AgentState) -> AgentState:
    """
    Classifies the user's question to decide which agent should handle it.
    This node acts as the supervisor.
    """
    print("--- 1. ROUTER NODE ---")
    question = state["original_question"]

    # Language detection and translation
    is_arabic = any('\u0600' <= ch <= '\u06FF' for ch in question)
    state["is_arabic"] = is_arabic
    if is_arabic:
        print("Translating from Arabic to English...")
        # Simplified translation call
        translation_response = llm.invoke(f"Translate the following financial query to professional English: '{question}'")
        processed_question = translation_response.content.strip()
    else:
        processed_question = question
    
    state["processed_question"] = processed_question
    print(f"Processed Question: {processed_question}")

    # Classification prompt
    classification_prompt = f"""You are a query classifier. Your job is to assign a user's financial query to ONE of the following categories:
    - portfolio: The user is asking about their personal portfolio, holdings, returns, P&L, or any data that would be in a database.
    - general: The user is asking a general financial question, about markets, stocks, or economic concepts that requires web search.
    - smalltalk: The user is engaging in casual conversation, greetings, or jokes.
    - identity: The user is asking about you, the AI assistant.
    - invalid: The query is offensive, gibberish, or completely out of context.

    Respond with ONLY the category name.
    User Query: "{processed_question}"
    Category:"""

    classification_response = llm.invoke(classification_prompt)
    classification = classification_response.content.strip().lower()
    print(f"Classification: {classification}")
    state["classification"] = classification
    return state

def portfolio_agent_node(state: AgentState) -> AgentState:
    """

    Handles database-related queries using Vanna.
    """
    print("--- 2a. PORTFOLIO AGENT ---")
    question = state["processed_question"]
    try:
        if not vn:
            raise ConnectionError("Vanna is not connected to the database.")

        sql = vn.generate_sql(question)
        state["sql_query"] = sql
        print(f"Generated SQL: {sql}")

        if not sql or "SELECT" not in sql.upper():
            raise ValueError("Failed to generate a valid SQL query.")

        df = vn.run_sql(sql)
        state["df_results"] = df

        if df is not None and not df.empty:
            explanation = vn.generate_explanation(sql=sql, dataframe=df.to_string(), question=question)
            state["final_answer"] = explanation
        else:
            state["final_answer"] = "I ran a query but found no data for your request."

    except Exception as e:
        print(f"Error in Portfolio Agent: {e}")
        traceback.print_exc()
        state["final_answer"] = "I'm sorry, I had trouble accessing the portfolio data. The query might be too complex or the data may not be available."

    return state

def internet_agent_node(state: AgentState) -> AgentState:
    """
    Handles general financial queries using Perplexity for web search.
    """
    print("--- 2b. INTERNET AGENT ---")
    question = state["processed_question"]
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3-sonar-large-32k-online",
        "messages": [
            {"role": "system", "content": "You are a helpful financial assistant. Answer precisely and factually."},
            {"role": "user", "content": question}
        ]
    }
    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content']
        state["final_answer"] = answer
    except Exception as e:
        print(f"Error in Internet Agent: {e}")
        state["final_answer"] = "Sorry, I couldn't access my online information source right now."

    return state

def smalltalk_agent_node(state: AgentState) -> AgentState:
    """Handles smalltalk and casual conversation."""
    print("--- 2c. SMALLTALK AGENT ---")
    question = state["processed_question"]
    prompt = f"You are a friendly financial assistant. Provide a brief, conversational response to: '{question}'"
    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state

def identity_agent_node(state: AgentState) -> AgentState:
    """Handles questions about the AI's identity."""
    print("--- 2d. IDENTITY AGENT ---")
    base_identity = "I am a multi-agent AI financial assistant, designed to help you analyze portfolio data and research market insights using specialized agents."
    state["final_answer"] = base_identity
    return state

def invalid_query_node(state: AgentState) -> AgentState:
    """Handles invalid or inappropriate queries."""
    print("--- 2e. INVALID QUERY ---")
    state["final_answer"] = "I can only answer questions related to finance and your portfolio. Please ask a relevant question."
    return state

def final_translation_node(state: AgentState) -> AgentState:
    """
    Translates the final answer back to Arabic if the original question was in Arabic.
    """
    print("--- 3. FINAL TRANSLATION NODE ---")
    if state["is_arabic"] and state["final_answer"]:
        print("Translating final answer to Arabic...")
        translation_prompt = f"Translate the following financial response to professional Arabic: '{state['final_answer']}'"
        response = llm.invoke(translation_prompt)
        state["final_answer"] = response.content
    return state

# --- Graph Definition ---

def decide_next_node(state: AgentState) -> str:
    """Conditional logic to route to the correct agent."""
    classification = state["classification"]
    if classification == "portfolio":
        return "portfolio_agent"
    elif classification == "general":
        return "internet_agent"
    elif classification == "smalltalk":
        return "smalltalk_agent"
    elif classification == "identity":
        return "identity_agent"
    else:
        return "invalid_query_agent"

# Create the StateGraph
builder = StateGraph(AgentState)

# Add nodes to the graph
builder.add_node("router", router_node)
builder.add_node("portfolio_agent", portfolio_agent_node)
builder.add_node("internet_agent", internet_agent_node)
builder.add_node("smalltalk_agent", smalltalk_agent_node)
builder.add_node("identity_agent", identity_agent_node)
builder.add_node("invalid_query_agent", invalid_query_node)
builder.add_node("final_translator", final_translation_node)

# Define the graph's flow
builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "portfolio_agent": "portfolio_agent",
        "internet_agent": "internet_agent",
        "smalltalk_agent": "smalltalk_agent",
        "identity_agent": "identity_agent",
        "invalid_query_agent": "invalid_query_agent",
    },
)

# All agent nodes lead to the final translator
builder.add_edge("portfolio_agent", "final_translator")
builder.add_edge("internet_agent", "final_translator")
builder.add_edge("smalltalk_agent", "final_translator")
builder.add_edge("identity_agent", "final_translator")
builder.add_edge("invalid_query_agent", "final_translator")

# The translator is the final step
builder.add_edge("final_translator", END)

# Compile the graph into a runnable object
agent_executor = builder.compile()

print("Agentic graph compiled successfully.")
