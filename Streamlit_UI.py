import streamlit as st
import streamlit.components.v1 as components
from markdown import markdown
import re

# Import the main agent executor from our agents file
from agents import agent_executor

# --- UI Configuration ---
st.set_page_config(page_title="📊 AI Financial Assistant", layout="centered")
st.title("📈 Multi-Agent Financial Assistant")
st.caption("I am an agentic AI that can access your portfolio database or browse the web. Ask me anything!")

# --- Helper Functions for UI Display ---
def display_formatted_response(title: str, text: str, html_height: int = 500):
    """
    Formats LLM output into an aesthetically pleasing HTML block with Markdown rendering.
    """
    if title:
        st.markdown(title)

    # Basic cleaning and markdown rendering
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    markdown_rendered_text = markdown(cleaned_text)

    # HTML wrapper for styling
    html_output = f"""
    <div style="
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #222;
    font-family: 'Segoe UI', sans-serif;
    margin-top: 1rem;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    overflow-y: auto;
    max-height: {html_height}px;
    ">
    {markdown_rendered_text}
    </div>
    """
    components.html(html_output, height=html_height + 20, scrolling=False)

# --- Main Application Logic ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle dictionary content for structured agent responses
        if isinstance(message["content"], dict):
            if "answer" in message["content"]:
                display_formatted_response("", message["content"]["answer"])
            if "sql_query" in message["content"] and message["content"]["sql_query"]:
                st.code(message["content"]["sql_query"], language="sql")
            if "dataframe" in message["content"] and message["content"]["dataframe"] is not None:
                st.dataframe(message["content"]["dataframe"])
        else:
            display_formatted_response("", message["content"])

# Get user input
question_input = st.chat_input("💬 Ask your question...")

if question_input:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": question_input})
    with st.chat_message("user"):
        display_formatted_response("", question_input)

    # Process the question with the agentic system
    with st.spinner("The agents are thinking..."):
        try:
            # The magic happens here: we invoke the LangGraph agent executor
            # It takes the user's question and the conversation history
            inputs = {
                "original_question": question_input,
                "chat_history": st.session_state.messages
            }
            final_state = agent_executor.invoke(inputs)

            # Extract the final answer and any artifacts (like SQL)
            response_content = {
                "answer": final_state.get("final_answer", "Sorry, I encountered an issue and couldn't find an answer."),
                "sql_query": final_state.get("sql_query"),
                "dataframe": final_state.get("df_results")
            }

            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            # Display AI response
            with st.chat_message("assistant"):
                display_formatted_response("", response_content["answer"])
                if response_content["sql_query"]:
                    st.code(response_content["sql_query"], language="sql")
                if response_content["dataframe"] is not None and not response_content["dataframe"].empty:
                    st.dataframe(response_content["dataframe"])

        except Exception as e:
            st.error(f"An error occurred in the agentic system: {e}")
            error_message = {"role": "assistant", "content": "I'm sorry, I ran into a critical error. Please try again."}
            st.session_state.messages.append(error_message)
            with st.chat_message("assistant"):
                st.write(error_message["content"])
