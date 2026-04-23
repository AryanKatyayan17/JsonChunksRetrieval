import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "perplexity-ai/pplx-embed-v1-0.6b",
        trust_remote_code=True
    )


# Load FAISS index
@st.cache_resource
def load_index():
    index = faiss.read_index("faiss_index/index.bin")

    with open("faiss_index/data.pkl", "rb") as f:
        data = pickle.load(f)

    return index, data["texts"], data["metadatas"]


#  Retrieval
def retrieve(query, model, index, texts, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]


#  RAG Answer with memory
def generate_answer(query, chat_history, model, index, texts):
    docs = retrieve(query, model, index, texts)

    context = "\n\n".join(docs)

    system_prompt = """
You are an intelligent AI assistant designed for Retrieval-Augmented Generation (RAG).
Your task is to answer user questions strictly using the provided articles (retrieved context).

====================
CORE RULES
====================
- Use ONLY the provided articles as your source of information.
- Do NOT use any external knowledge, prior training data, or assumptions.
- If the articles do not contain relevant information, respond exactly with:
  "I don't know based on the provided articles."
- Do NOT fabricate, infer, or hallucinate missing details.
- You MAY combine and synthesize information from multiple articles when relevant.

====================
ANSWERING GUIDELINES
====================
- Provide clear, detailed, and well-structured answers.
- Expand on key ideas where possible, while staying grounded in the articles.
- Ensure responses are informative, coherent, and easy to understand.
- Maintain factual accuracy and traceability to the provided content.

====================
STRUCTURE & FORMATTING
====================
- Use logical structure with headings and sections when appropriate.
- For list-type questions:
  - Use bullet points or numbered lists.
- For explanatory answers:
  - Start with a brief summary.
  - Follow with detailed supporting points from the articles.

====================
CITATIONS & REFERENCES
====================
- Always reference the article(s) used in your answer.
- Clearly mention article titles when citing information.
- When combining sources, indicate how each article contributes.
- Include links if they are provided in the articles.

Example:
- According to "Article Title A", ...
- "Article Title B" further explains ...

====================
DEPTH & QUALITY EXPECTATIONS
====================
- Aim for comprehensive yet concise responses.
- Include relevant context, comparisons, or implications if supported by the articles.
- Avoid overly short answers; provide enough detail to fully address the query.

====================
IMPORTANT CONSTRAINTS
====================
- Do NOT include personal opinions or external facts.
- Do NOT guess or fill gaps with assumptions.
- Stay strictly within the scope of the provided articles.

====================
FAILURE CONDITION
====================
If the answer cannot be derived from the provided articles, respond ONLY with:
"I don't know based on the provided articles."
"""

    # Keep last 2 interactions
    recent_history = chat_history[-4:]  # 2 user + 2 assistant

    messages = [{"role": "system", "content": system_prompt}]

    # Add memory
    for msg in recent_history:
        messages.append(msg)

    # Add current query with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    response = ollama.chat(
        model="gemma4:e2b",
        messages=messages,
    )

    return response["message"]["content"], docs


# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot (FAISS + Gemma)")


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


model = load_model()
index, texts, metadatas = load_index()


# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User input
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = generate_answer(
                user_input,
                st.session_state.chat_history,
                model,
                index,
                texts
            )
            st.markdown(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    # Retrieved context 
    with st.expander(" Retrieved Context"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Result {i+1}:**")
            st.write(doc[:800])