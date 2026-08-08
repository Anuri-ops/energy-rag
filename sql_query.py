"""
sql_query.py — the natural-language-to-SQL piece.
Turns a plain-English data question into a SQL query, runs it, and answers.
"""
import os
import sqlite3
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "data/operational_data.csv"
TABLE_NAME = "operations"

def get_connection():
    """Load the CSV into an in-memory SQLite database and return the connection."""
    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(":memory:")
    df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
    return conn

def get_schema():
    """Describe the table so the LLM knows what columns it can query."""
    df = pd.read_csv(CSV_PATH)
    cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    return f"Table '{TABLE_NAME}' with columns: {cols}"

def answer_data_question(question):
    schema = get_schema()

    # 1. Ask the LLM to write a SQL query for the question
    sql_prompt = (
        f"You are a SQL expert. Given this table:\n{schema}\n\n"
        f"Write a single SQLite SELECT query that answers this question:\n{question}\n\n"
        "Return ONLY the SQL query, no explanation, no markdown formatting."
    )
    sql_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0,
    )
    sql_query = sql_response.choices[0].message.content.strip()
    # strip markdown fences if the model added them
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # 2. SAFETY: only allow SELECT queries
    if not sql_query.lower().startswith("select"):
        return "I can only run read-only data queries, and that request wasn't a valid one."

    # 3. Run the query
    try:
        conn = get_connection()
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
    except Exception as e:
        return f"I couldn't run that query. ({e})"

    # 4. Ask the LLM to phrase a plain-English answer from the results
    answer_prompt = (
        f"Question: {question}\n\n"
        f"SQL query used: {sql_query}\n\n"
        f"Query result:\n{result_df.to_string(index=False)}\n\n"
        "Answer the question in plain English based on this result. Be concise."
    )
    answer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
    )
    return answer_response.choices[0].message.content

# Quick test when run directly
if __name__ == "__main__":
    q = "How many assets had a FAULT status?"
    print("Q:", q)
    print("A:", answer_data_question(q))