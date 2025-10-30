# Multi-Agent Financial AI Assistant

This project implements a sophisticated, multi-agent financial chatbot using Streamlit for the user interface and LangGraph for the agentic backend. The system is designed to intelligently handle a variety of user queries by routing them to specialized agents, each an expert in its domain.

## Agentic Architecture

The core of this application is a stateful graph of agents orchestrated by LangGraph. When a user asks a question, it triggers a series of steps within this graph.


*(A simple diagram illustrating the flow)*

### 1. The Supervisor/Router Agent (`router_node`)
This is the entry point and the "brain" of the operation. Its responsibilities are:
- **Language Detection:** It first checks if the user's query is in Arabic.
- **Translation:** If the query is in Arabic, it translates it to English to ensure consistent processing by all downstream agents.
- **Intent Classification:** It uses a powerful language model (LLM) to classify the user's intent into one of several categories:
  - `portfolio`: Questions about personal investment data.
  - `general`: General financial or market questions.
  - `smalltalk`: Casual conversation.
  - `identity`: Questions about the AI itself.
  - `invalid`: Offensive, nonsensical, or out-of-scope queries.
- **Routing:** Based on the classification, it directs the query to the appropriate specialist agent.

### 2. Specialist Agents
Each specialist agent is a node in the graph designed to perform a specific task with high proficiency.

#### a. Portfolio Agent (`portfolio_agent_node`)
- **Domain:** Manages all interactions with the user's portfolio database.
- **Tools:** Uses the **Vanna** library.
- **Workflow:**
    1.  Receives the user's question.
    2.  Generates a SQL query using its trained knowledge of the database schema, documentation, and example questions.
    3.  Executes the SQL query against the live PostgreSQL database.
    4.  Receives the data as a Pandas DataFrame.
    5.  Generates a clear, natural language explanation of the results, tailored for a non-technical audience.
    6.  Handles errors gracefully if SQL generation fails or the query returns no data.

#### b. Internet Agent (`internet_agent_node`)
- **Domain:** Answers general financial questions that require up-to-date, real-world information.
- **Tools:** Uses the **Perplexity API**, which has real-time web search capabilities.
- **Workflow:**
    1.  Receives a question like "What is the current outlook for the tech sector?" or "Explain quantitative easing."
    2.  Sends the query to the Perplexity API.
    3.  Returns a factual, sourced answer based on current information from the web.

#### c. Smalltalk & Identity Agents
- **Domain:** Manages conversational and meta-questions.
- **Tools:** Utilizes the base LLM with specialized system prompts.
- **Workflow:**
    - The `smalltalk_agent` provides brief, friendly responses to greetings and casual chat.
    - The `identity_agent` provides a consistent explanation of its capabilities as a multi-agent system.

### 3. Final Translation Node
- **Responsibility:** Ensures a seamless multilingual experience.
- **Workflow:** After a specialist agent has generated a final answer in English, this node checks if the original user query was in Arabic. If so, it translates the final response back into professional Arabic before presenting it to the user.

## File Structure
The project is now organized into modular, single-responsibility files:

- **`streamlit_ui.py`**: Contains only the code for rendering the web interface. It's the "view" in our application.
- **`agents.py`**: The core logic engine. It defines the `AgentState`, all agent nodes, and the LangGraph structure that connects them. It's the "controller" and "model".
- **`vanna_util.py`**: A utility module for configuring, connecting, and training the Vanna instance. This isolates database-specific setup.
- **`.streamlit/secrets.toml`**: A configuration file (not included in repo) to securely store API keys and database credentials.

This agentic approach makes the system more robust, easier to debug, and highly extensible. New tools or agents (e.g., a "Trading Execution Agent" or a "Report Generation Agent") can be added as new nodes to the graph with minimal changes to the existing logic.
