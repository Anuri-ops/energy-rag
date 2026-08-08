"""
app.py — Streamlit demo UI.
Routes questions through the orchestrator (RAG for documents, NL-to-SQL for data).
Builds the vector index on first startup if needed. Per-session rate limiting.
Run locally:  streamlit run app.py
"""
import os
import streamlit as st

# --- Build the RAG index on startup if it's not already there ---
@st.cache_resource
def ensure_index():
    import chromadb
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="energy_docs")
    if collection.count() > 0:
        return True
    DOCS_DIR = "data/docs"
    def chunk_text(text, size=500, overlap=50):
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += size - overlap
        return chunks
    def embed(text):
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding
    doc_id = 0
    for filename in os.listdir(DOCS_DIR):
        with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text):
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embed(chunk)],
                documents=[chunk],
                metadatas=[{"source": filename}],
            )
            doc_id += 1
    return True

MAX_QUESTIONS_PER_SESSION = 10

st.set_page_config(page_title="Energy Ops Assistant", page_icon="⚡")
st.title("⚡ Energy Operations Assistant")
st.write(
    "Ask a plain-language question about energy asset operations. "
    "The assistant automatically decides whether to answer from the **documents** "
    "(procedures, manuals) or from the **operational data** (asset readings), and shows which it used."
)

with st.spinner("Preparing knowledge base..."):
    ensure_index()

# Route through the orchestrator (RAG + NL-to-SQL under one router)
from orchestrator import assistant

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

with st.expander("Example questions"):
    st.markdown(
        "**Document questions**\n"
        "- At what gearbox oil temperature must a turbine be shut down?\n"
        "- What triggers an emergency shutdown?\n"
        "- When should I *not* send a field team for an output drop?\n\n"
        "**Data questions**\n"
        "- How many assets had a FAULT status?\n"
        "- What was the average power output for Site3_Solar?\n"
        "- Which asset had the most anomalies?"
    )

remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
st.caption(f"Demo limit: {remaining} question(s) remaining this session.")

question = st.text_input("Your question:")

if question:
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            "You've reached the demo question limit for this session. "
            "This cap simply protects the demo from overuse. Thanks for trying it!"
        )
    else:
        with st.spinner("Routing and answering..."):
            answer = assistant(question)
        st.session_state.question_count += 1
        st.markdown("### Answer")
        st.write(answer)

st.divider()
st.caption(
    "Built with Python, OpenAI, Chroma, and Streamlit. Routes between RAG (documents) "
    "and natural-language-to-SQL (operational data). Data is synthetic. A demonstration of "
    "retrieval-augmented generation and agentic routing."
)
