import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up Chroma — a local vector database stored in the ./chroma_db folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="energy_docs")

DOCS_DIR = "data/docs"

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks of roughly chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap keeps context from breaking across chunks
    return chunks

def embed(text):
    """Turn a piece of text into a vector using OpenAI's embedding model."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Go through every document, chunk it, embed each chunk, and store it
doc_id = 0
for filename in os.listdir(DOCS_DIR):
    filepath = os.path.join(DOCS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    for chunk in chunks:
        vector = embed(chunk)
        collection.add(
            ids=[f"doc_{doc_id}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": filename}]  # remember which file it came from
        )
        doc_id += 1
    print(f"Ingested {filename} ({len(chunks)} chunks)")

print(f"\nDone. {doc_id} chunks stored in Chroma.")