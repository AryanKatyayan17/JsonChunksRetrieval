import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "perplexity-ai/pplx-embed-v1-0.6b",
        trust_remote_code=True
    )

# Load FAISS index + data
@st.cache_resource
def load_index():
    index = faiss.read_index("faiss_index/index.bin")

    with open("faiss_index/data.pkl", "rb") as f:
        data = pickle.load(f)

    return index, data["texts"], data["metadatas"]

# Retrieval
def retrieve(query, model, index, texts, top_k=3):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    return [texts[i] for i in indices[0]]


# RAG Answer
def generate_answer(query, model, index, texts):
    docs = retrieve(query, model, index, texts)

    context = "\n\n".join(docs)

    system_prompt = """
    You are an intelligent AI assistant that answers questions using the provided articles.

        Rules:
        - Use ONLY the provided articles as your source of information
        - Do NOT use any external knowledge
        - If the articles do not contain relevant information, respond with:
            "I don't know based on the provided articles."
        - Do NOT attempt to answer using general knowledge
        - You may combine information from multiple articles if relevant
        - Be clear, structured, and helpful

        Answer Guidelines:
        - For list-type questions, return bullet points or numbered lists
        - For explanations, keep them concise but informative
        - Mention article titles when relevant
        - Include links when useful for reference
    """

    response = ollama.chat(
        model="gemma4:e2b",  # lightweight model
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
    )

    return response["message"]["content"], docs


# UI
st.set_page_config(page_title="RAG App", layout="wide")

st.title(" RAG Search with FAISS + Gemma")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        model = load_model()
        index, texts, metadatas = load_index()

        answer, docs = generate_answer(query, model, index, texts)

    st.subheader("Answer")
    st.write(answer)

    with st.expander(" Retrieved Context"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Result {i+1}:**")
            st.write(doc[:1000])