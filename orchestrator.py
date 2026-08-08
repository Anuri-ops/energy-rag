"""
orchestrator.py — the router.
Decides whether a question is about DOCUMENTS (use RAG) or DATA (use SQL),
then sends it to the right engine.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

from query import answer_question          # RAG (documents)
from sql_query import answer_data_question # NL-to-SQL (operational data)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def route(question):
    """Ask the LLM to classify the question as 'documents' or 'data'."""
    routing_prompt = (
        "Classify the following question into exactly one category:\n"
        "- 'data': questions about operational figures, counts, averages, statuses, "
        "specific assets' output, anomalies, or anything answerable from a table of readings.\n"
        "- 'documents': questions about procedures, thresholds, guidelines, causes, "
        "safety, maintenance, or how/why something is done.\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY one word: data or documents."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": routing_prompt}],
        temperature=0,
    )
    category = response.choices[0].message.content.strip().lower()
    return "data" if "data" in category else "documents"

def assistant(question):
    """The single entry point: route, then answer with the right engine."""
    category = route(question)
    if category == "data":
        answer = answer_data_question(question)
        return f"[Answered from operational data]\n\n{answer}"
    else:
        answer = answer_question(question)
        return f"[Answered from documents]\n\n{answer}"

# Quick test — one of each type
if __name__ == "__main__":
    for q in [
        "At what gearbox oil temperature must a turbine be shut down?",  # documents
        "How many assets had a FAULT status?",                          # data
    ]:
        print("Q:", q)
        print(assistant(q))
        print("-" * 50)