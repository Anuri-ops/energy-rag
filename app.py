"""
app.py — Streamlit demo UI with per-session rate limiting.
Builds the vector index on first startup if it doesn't exist (so it works on a fresh deploy).
Run locally:  streamlit run app.py
"""
import os
import streamlit as st

# --- Build the index on startup if it's not already there ---
# On a fresh Streamlit deploy the chroma_db folder doesn't exist (it's gitignored),
# so we ingest the documents once, the first time the app boots.
@st.cache_resource
def ensure_index():
    import chromadb
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="energy_docs")

    # If already populated, do nothing
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
    "Answers are retrieved from the document knowledge base and cite their source."
)

# Make sure the index exists before we allow questions
with st.spinner("Preparing knowledge base..."):
    ensure_index()

# Import after index is ready
from query import answer_question

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

with st.expander("Example questions"):
    st.markdown(
        "- At what gearbox oil temperature must a turbine be shut down?\n"
        "- What caused the output drop at Site 3?\n"
        "- When should I *not* send a field team for an output drop?\n"
        "- What triggers an emergency shutdown?"
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
        with st.spinner("Retrieving and answering..."):
            answer = answer_question(question)
        st.session_state.question_count += 1
        st.markdown("### Answer")
        st.write(answer)

st.divider()
st.caption(
    "Built with Python, OpenAI embeddings, Chroma, and Streamlit. "
    "Data is synthetic. This is a demonstration of RAG fundamentals."
)
