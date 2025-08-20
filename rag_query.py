from mistralai import Mistral
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer


load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("❌ Pas de clé API MISTRAL trouvée. Vérifie ton .env")

llm_client = Mistral(api_key=api_key)

client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_collection("energie_rag")

embedder = SentenceTransformer("all-MiniLM-L6-v2") #Le même embedder que dans ingest.py

def rag_query(question: str, k: int = 3):
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=k)

    documents = results["documents"][0]
    sources = results["metadatas"][0]
    context = "\n\n".join(documents)

    prompt = f"""
    Tu es un assistant expert en énergie.
    Utilise les passages suivants pour répondre à la question de manière concise et claire.

    Contexte :
    {context}

    Question :
    {question}

    Réponse :
    """

    resp = llm_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = resp.choices[0].message.content
    return answer, documents, sources



if __name__ == "__main__":
    question = "Comment est réparti le bouquet énergétique français ?"
    answer, docs, sources = rag_query(question)

    print("🔎 Question :", question)
    print("\n📚 Passages retenus :")
    for d in docs:
        print("-", d[:200], "...")
    print("\n📌 Réponse générée :\n", answer)
