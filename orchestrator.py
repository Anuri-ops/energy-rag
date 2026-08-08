"""
orchestrator.py — the router (Option 2 + query decomposition).
Decides whether a question needs DOCUMENTS (RAG), DATA (SQL), or BOTH.
For BOTH, it DECOMPOSES the question into a data sub-question and a document
sub-question, sends each engine its focused part, then combines the results.
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from query import answer_question          # RAG (documents)
from sql_query import answer_data_question # NL-to-SQL (operational data)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def route(question):
    """Classify the question as 'data', 'documents', or 'both'."""
    routing_prompt = (
        "Classify the following question into exactly one category:\n"
        "- 'data': answerable ONLY from a table of operational readings "
        "(counts, averages, statuses, specific assets' output, anomalies).\n"
        "- 'documents': answerable ONLY from procedures, manuals, guidelines, "
        "thresholds, causes, safety or maintenance text.\n"
        "- 'both': needs BOTH the operational data AND the documents "
        "(e.g. 'which assets faulted, and what should I do about it?').\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY one word: data, documents, or both."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": routing_prompt}],
        temperature=0,
    )
    category = response.choices[0].message.content.strip().lower()
    if "both" in category:
        return "both"
    if "data" in category:
        return "data"
    return "documents"

def decompose(question):
    """Split a 'both' question into a focused data sub-question and a document sub-question."""
    decompose_prompt = (
        "This question needs both operational data and reference documents. "
        "Split it into two focused sub-questions.\n\n"
        f"Question: {question}\n\n"
        'Reply as JSON only: {"data_question": "...", "document_question": "..."}\n'
        "The data_question should ask only about the operational figures/statuses. "
        "The document_question should ask only about the procedure/guidance/cause."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": decompose_prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        parts = json.loads(raw)
        return parts["data_question"], parts["document_question"]
    except Exception:
        # fallback: if parsing fails, send the whole question to both
        return question, question

def combine(question, doc_answer, data_answer):
    """Synthesise one unified answer from both engines' results."""
    combine_prompt = (
        f"Question: {question}\n\n"
        f"Answer from the documents:\n{doc_answer}\n\n"
        f"Answer from the operational data:\n{data_answer}\n\n"
        "Write ONE clear, unified answer to the question that combines both sources. "
        "Keep the source citations where relevant. Be concise."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": combine_prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

def assistant(question):
    """Single entry point: route, then answer with the right engine(s)."""
    category = route(question)

    if category == "data":
        answer = answer_data_question(question)
        return f"[Answered from operational data]\n\n{answer}"

    if category == "documents":
        answer = answer_question(question)
        return f"[Answered from documents]\n\n{answer}"

    # category == "both" -> decompose, run each engine on its focused part, then combine
    data_q, doc_q = decompose(question)
    data_answer = answer_data_question(data_q)
    doc_answer = answer_question(doc_q)
    unified = combine(question, doc_answer, data_answer)
    return f"[Answered from documents + operational data]\n\n{unified}"

# Quick test — one of each type
if __name__ == "__main__":
    tests = [
        "At what gearbox oil temperature must a turbine be shut down?",  # documents
        "How many assets had a FAULT status?",                          # data
        "Which assets had a FAULT, and what should I do about a fault?", # both
    ]
    for q in tests:
        print("Q:", q)
        print(assistant(q))
        print("-" * 60)
