"""
query.py — the RAG brain.
Flow: embed question -> retrieve closest chunks -> grounded prompt -> LLM answer.
"""
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
# get_or_create so it never crashes if the collection isn't built yet
collection = chroma_client.get_or_create_collection(name="energy_docs")

def embed(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def answer_question(question, n_results=3):
    q_vec = embed(question)
    results = collection.query(query_embeddings=[q_vec], n_results=n_results)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources))

    system_prompt = (
        "You are an energy operations assistant. Answer the question using ONLY the "
        "context provided below. If the answer is not in the context, say you don't "
        "have that information. Always cite the source file(s) you used."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    q = "At what gearbox oil temperature must a turbine be shut down?"
    print("Q:", q)
    print("A:", answer_question(q))
