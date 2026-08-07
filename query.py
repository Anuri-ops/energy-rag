import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load the key and set up the clients
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="energy_docs")

def embed(text):
    """Same embedding model used in ingest — the question must live in the same vector space."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def answer_question(question, n_results=3):
    # 1. Embed the question
    question_vector = embed(question)

    # 2. Retrieve the most relevant chunks from Chroma
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=n_results
    )
    retrieved_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # 3. Build the context block from retrieved chunks
    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for chunk, src in zip(retrieved_chunks, sources)
    )

    # 4. Build the prompt with strict grounding instructions
    system_prompt = (
        "You are an energy operations assistant. Answer the question using ONLY the "
        "context provided below. If the answer is not in the context, say you don't "
        "have that information. Always cite the source file(s) you used."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # 5. Ask the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # low temperature = factual, consistent answers
    )
    return response.choices[0].message.content

# Quick test when run directly
if __name__ == "__main__":
    q = "At what gearbox oil temperature must a turbine be shut down?"
    print("Q:", q)
    print("A:", answer_question(q))