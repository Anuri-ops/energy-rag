"""
app.py — Streamlit demo UI with simple per-session rate limiting.
Run locally:  streamlit run app.py
"""
import streamlit as st
from query import answer_question

# --- simple protection for a public, keyed demo ---
MAX_QUESTIONS_PER_SESSION = 10   # a visitor can ask this many, then it stops

st.set_page_config(page_title="Energy Ops Assistant", page_icon="⚡")

st.title("⚡ Energy Operations Assistant")
st.write(
    "Ask a plain-language question about energy asset operations. "
    "Answers are retrieved from the document knowledge base and cite their source."
)

# Track how many questions this browser session has asked
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

with st.expander("Example questions"):
    st.markdown(
        "- At what gearbox oil temperature must a turbine be shut down?\n"
        "- What caused the output drop at Site 3?\n"
        "- When should I *not* send a field team for an output drop?\n"
        "- What triggers an emergency shutdown?"
    )

# Show remaining questions so it's transparent
remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
st.caption(f"Demo limit: {remaining} question(s) remaining this session.")

question = st.text_input("Your question:")

if question:
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            "You've reached the demo question limit for this session. "
            "Refresh isn't required to review previous answers — this cap simply "
            "protects the demo from overuse. Thanks for trying it!"
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
