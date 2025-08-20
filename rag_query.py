# rag_query.py
from mistralai import Mistral
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ===== 1) Charger variables d'environnement =====
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("❌ Pas de clé API MISTRAL trouvée. Vérifie ton .env")

# ===== 2) Initialiser client Mistral =====
llm_client = Mistral(api_key=api_key)

# ===== 3) Charger l’index Chroma =====
client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_collection("energie_rag")

# Embeddings doivent être les mêmes que dans ingest.py
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===== 4) Fonction RAG =====
def rag_query(question: str, k: int = 3):
    # Encoder la question
    q_embedding = embedder.encode(question).tolist()

    # Récupérer les chunks les plus proches
    results = collection.query(query_embeddings=[q_embedding], n_results=k)

    documents = results["documents"][0]
    sources = results["metadatas"][0]

    # Contexte combiné
    context = "\n\n".join(documents)

    # Prompt pour Mistral
    prompt = f"""
    Tu es un assistant expert en énergie.
    Utilise les passages suivants pour répondre à la question de manière concise et claire.

    Contexte :
    {context}

    Question :
    {question}

    Réponse :
    """

    # Appel au LLM
    resp = llm_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = resp.choices[0].message.content
    return answer, documents, sources

# ===== 5) Exemple =====
if __name__ == "__main__":
    question = "Comment est réparti le bouquet énergétique français ?"
    answer, docs, sources = rag_query(question)

    print("🔎 Question :", question)
    print("\n📚 Passages retenus :")
    for d in docs:
        print("-", d[:200], "...")
    print("\n📌 Réponse générée :\n", answer)
