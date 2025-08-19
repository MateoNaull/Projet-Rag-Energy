# src/rag_query.py

import chromadb

# ========= 1) Connexion à Chroma =========
client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_or_create_collection("energie_rag")

# ========= 2) Fonction de requête =========
def query_index(question: str, top_k: int = 3):
    """Interroge l'index et retourne les passages les plus pertinents"""
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    return results


# ========= 3) Exemple d’utilisation =========
if __name__ == "__main__":
    question = "Quelle est la consommation énergétique en Île-de-France ?"
    results = query_index(question, top_k=3)

    print("\n🔎 Question :", question)
    print("📌 Résultats pertinents :\n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"--- Passage {i+1} ---")
        print(doc)
        print("Source:", results["metadatas"][0][i]["source"])
        print()
