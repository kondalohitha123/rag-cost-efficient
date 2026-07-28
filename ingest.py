import os
import json
import hashlib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load a lightweight, free, local embedding model (runs entirely offline)
print("Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

DB_FILE = "vector_store.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db_data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db_data, f, indent=2)

def pure_python_split(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += (chunk_size - chunk_overlap)
        
    return chunks

def compute_hash(text: str, source: str) -> str:
    """Computes SHA-256 hash to enforce idempotency."""
    return hashlib.sha256(f"{source}:{text}".encode('utf-8')).hexdigest()

def get_embedding(text: str) -> list[float]:
    # Generates 384-dimensional dense vectors locally
    embedding = model.encode(text)
    return embedding.tolist()

def ingest_file(file_path: str, category: str = "general", chunk_size: int = 300, chunk_overlap: int = 50):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return

    print(f"Reading file: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = pure_python_split(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    db_data = load_db()

    added_count = 0
    for chunk in chunks:
        chunk_id = compute_hash(chunk, file_path)
        
        # Idempotency check: only insert new unique chunks
        if chunk_id not in db_data:
            print(f"Generating local embedding for chunk...")
            embedding = get_embedding(chunk)
            db_data[chunk_id] = {
                "id": chunk_id,
                "text": chunk,
                "vector": embedding,
                "source": file_path,
                "category": category
            }
            added_count += 1

    save_db(db_data)
    print(f"Successfully processed {len(chunks)} chunk(s). {added_count} new chunk(s) added to '{DB_FILE}'.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(base_dir, "data", "sample.md")
    
    ingest_file(sample_file, category="policy")