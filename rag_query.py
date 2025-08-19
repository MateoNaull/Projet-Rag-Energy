# src/rag_query.py

import chromadb
from transformers import pipeline

# ========== 1) Connexion à Chroma ==========
client_chroma = chromadb.PersistentClient(path="outputs/index")
collection = client_chroma.get_or_create_collection("energie_rag")

# ========== 2) Charger un modèle Hugging Face ==========
# Flan-T5-base : petit modèle d'instruction, tourne sur CPU
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# ========== 3) Recherche dans l'index ==========
def query_index(question: str, top_k: int = 3):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    return results

# ========== 4) Génération de réponse ==========
def summarize_passage(text, max_tokens=100):
    result = generator(
        f"Résume en quelques phrases le texte suivant : {text}",
        max_new_tokens=max_tokens
    )
    return result[0]["generated_text"]

def generate_answer(question, docs):
    # Résumer chaque passage
    summaries = [summarize_passage(doc) for doc in docs]
    context = "\n\n".join(summaries)

    prompt = f"""
    Tu es un assistant spécialisé en énergie.
    Question : {question}

    Contexte (résumés des documents) :
    {context}

    Réponds uniquement à partir du contexte fourni, de façon claire et concise.
    """
    result = generator(prompt, max_new_tokens=256)
    return result[0]["generated_text"]


# ========== 5) Exemple d’utilisation ==========
if __name__ == "__main__":
    question = "Quelle est la consommation énergétique en Île-de-France ?"
    results = query_index(question, top_k=3)

    docs = results["documents"][0]

    print("\n🔎 Question :", question)

    print("\n📌 Réponse générée :\n")
    answer = generate_answer(question, docs)
    print(answer)
