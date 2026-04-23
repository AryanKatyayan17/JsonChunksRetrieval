import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# Load embedding model (same as indexing)
model = SentenceTransformer(
    "perplexity-ai/pplx-embed-v1-0.6b",
    trust_remote_code=True
)


def load_index():
    index = faiss.read_index("faiss_index/index.bin")

    with open("faiss_index/data.pkl", "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    metadatas = data["metadatas"]

    return index, texts, metadatas


def retrieve(query, top_k=5):
    index, texts, _ = load_index()

    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = [texts[i] for i in indices[0]]

    return results


def generate_answer(query):
    docs = retrieve(query)

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
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    query = "What is Artemis II mission?"

    answer = generate_answer(query)

    print("\nAnswer:\n")
    print(answer)