# vanna_util.py

import streamlit as st
from typing import Optional, Dict, Any

# Vanna specific imports
from vanna.ollama import Ollama
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, Ollama):
    """Custom Vanna class for combining Vector Store and LLM."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

@st.cache_resource
def get_vanna_instance():
    """
    Creates, configures, and trains a Vanna instance.
    Uses Streamlit's cache to avoid re-initializing and re-training on every run.
    """
    # --- Vanna Configuration from Streamlit Secrets ---
    OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", "llama3")
    OLLAMA_API_URL = st.secrets.get("OLLAMA_API_URL", "http://localhost:11434")

    config = {
        'model': OLLAMA_MODEL,
        'ollama_api_url': OLLAMA_API_URL,
        'persist_directory': './vanna_chroma_store'
    }

    vn = MyVanna(config=config)

    # --- Database Connection ---
    try:
        vn.connect_to_postgres(
            host=st.secrets.get("DB_HOST"),
            dbname=st.secrets.get("DB_NAME"),
            user=st.secrets.get("DB_USER"),
            password=st.secrets.get("DB_PASSWORD"),
            port=st.secrets.get("DB_PORT")
        )
        print("Successfully connected to Vanna database.")
    except Exception as e:
        st.error(f"Failed to connect to Vanna database: {e}")
        return None # Return None if connection fails

    # --- Train the Vanna Model ---
    # This is a good place to check if training has already been done
    # For simplicity here, we train every time the cache is cleared.
    print("Training Vanna instance with documentation and Q&A pairs...")
    train_vanna(vn)
    print("Vanna training complete.")

    return vn

def train_vanna(vn: MyVanna):
    """
    Populates the Vanna instance with documentation, DDL, and question-SQL pairs.
    This function should contain all your vn.add_* calls.
    """
    # You can load DDL from a file
    # with open('ddl.sql', 'r') as f:
    #     sql = f.read()
    # vn.add_ddl(ddl=sql)

    # --- Add Documentation (Table and Column level) ---
    vn.add_documentation(
        table_name='portfolio_summary',
        schema_name='ai_trading',
        documentation='Provides aggregated summary metrics for each investment portfolio, including various time-based returns, profit figures, and net liquidity.'
    )
    vn.add_documentation(
        table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_return',
        documentation='The Year-To-Date (YTD) percentage return of the portfolio. Key metric for annual performance.'
    )
    vn.add_documentation(
        table_name='portfolio_summary', schema_name='ai_trading', column_name='ytd_profit',
        documentation='The Year-To-Date (YTD) monetary profit (P&L) of the portfolio.'
    )
    # ... (Add all other vn.add_documentation calls here) ...
    vn.add_documentation(
        table_name='portfolio_holdings_realized_pnl',
        schema_name='ai_trading',
        documentation='This table provides detailed profit and loss (P&L) information for portfolio holdings, including unrealized and realized gains/losses.'
    )
    # ... etc.

    # --- Add Question-SQL Pairs ---
    vn.add_question_sql(
        question='What is the best performing portfolio by YTD return?',
        sql="SELECT portfolio_name, ytd_return FROM ai_trading.portfolio_summary ORDER BY ytd_return DESC LIMIT 1;"
    )
    vn.add_question_sql(
        question='Which portfolio has the highest all-time profit?',
        sql="SELECT portfolio_name, all_profit FROM ai_trading.portfolio_summary ORDER BY all_profit DESC LIMIT 1;"
    )
    vn.add_question_sql(
        question='Show me my realized gains this month.',
        sql="SELECT SUM(daily_realized_pnl) AS realized_gains_this_month FROM ai_trading.portfolio_holdings_realized_pnl WHERE datetime >= date_trunc('month', current_date) AND datetime < date_trunc('month', current_date) + interval '1 month';"
    )
    # ... (Add all other vn.add_question_sql calls here) ...

    # It's also a good practice to train on information schema
    # df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
    # vn.train(df_information_schema=df_information_schema)
