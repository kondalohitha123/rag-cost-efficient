import os
import json
import math
import ollama
from sentence_transformers import SentenceTransformer

# Load local embedding model
print("Loading local embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

DB_FILE = "vector_store.json"

def load_db():
    if not os.path.exists(DB_FILE):
        print(f"Error: {DB_FILE} not found. Please run ingest.py first.")
        return {}
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def retrieve_context(query_text: str, top_k: int = 1) -> str:
    db_data = load_db()
    if not db_data:
        return ""

    query_vector = model.encode(query_text).tolist()
    results = []

    for chunk_id, data in db_data.items():
        sim_score = cosine_similarity(query_vector, data["vector"])
        results.append({
            "text": data["text"],
            "score": sim_score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return "\n\n".join([res["text"] for res in results[:top_k]])

def generate_answer(query_text: str):
    context = retrieve_context(query_text)
    
    if not context:
        print("No relevant context found in vector store.")
        return

    # Prompt sent to local Llama 3.2
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.

Context:
{context}

Question: {query_text}
Answer:"""

    print("\nGenerating final answer using Llama 3.2...")
    
    # Call local Ollama LLM
    response = ollama.generate(model="llama3.2", prompt=prompt)
    
    print("\n--- Final Generated Answer ---")
    print(response['response'])

if __name__ == "__main__":
    user_query = "What is the policy on leave and work hours?"
    generate_answer(user_query)