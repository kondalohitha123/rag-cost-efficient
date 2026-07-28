import os
import json
import math
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# Page setup
st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖")
st.title("🤖 Local Document Assistant")
st.caption("Powered by all-MiniLM-L6-v2 & Llama 3.2 via Ollama")

DB_FILE = "vector_store.json"

# Load embedding model once into cache so it runs fast
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def load_db():
    if not os.path.exists(DB_FILE):
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

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User chat input box
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            context = retrieve_context(prompt)
            
            if not context:
                response_text = "No relevant context found in vector store."
            else:
                # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            context = retrieve_context(prompt)
            
            if not context:
                response_text = "No relevant context found in vector store."
            else:
                # PASTE THE NEW PROMPT HERE:
                rag_prompt = f"""You are a helpful assistant. Answer the user's question. 

Use the provided Context if it contains relevant information to answer the question. 
If the Context is not relevant or does not contain the answer, answer the question using your general knowledge.

Context:
{context}

Question: {prompt}
Answer:"""

                response = ollama.generate(model="llama3.2", prompt=rag_prompt)
                response_text = response['response']

            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})