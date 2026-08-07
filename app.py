import streamlit as st
from query import answer_question

# Page setup
st.set_page_config(page_title="Energy Ops Assistant", page_icon="⚡")

st.title("⚡ Energy Operations Assistant")
st.write(
    "Ask a plain-language question about energy asset operations. "
    "Answers are retrieved from the document knowledge base and cite their source."
)

# A few example questions to guide the demo
with st.expander("Example questions"):
    st.markdown(
        "- At what gearbox oil temperature must a turbine be shut down?\n"
        "- What caused the output drop at Site 3?\n"
        "- When should I *not* send a field team for an output drop?\n"
        "- What triggers an emergency shutdown?"
    )

# The input box
question = st.text_input("Your question:")

# When the user submits, run RAG and show the answer
if question:
    with st.spinner("Retrieving and answering..."):
        answer = answer_question(question)
    st.markdown("### Answer")
    st.write(answer)