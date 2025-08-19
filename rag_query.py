# src/rag_query.py

import chromadb
from transformers import AutoTokenizer, pipeline

# ===== 0) Modèle HuggingFace =====
MODEL_ID = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generator = pipeline("text2text-generation", model=MODEL_ID, tokenizer=tokenizer)

MAX_INPUT_TOKENS = 480  # sécurité sous 512

def truncate_to_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """Coupe un texte pour qu'il tienne dans la limite du modèle"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

# ===== 1) Connexion à Chroma =====
client_chroma = chromadb.PersistentClient(path="outputs/index")
collection = client_chroma.get_or_create_collection("energie_rag")

def query_index(question: str, top_k: int = 3):
    return collection.query(query_texts=[question], n_results=top_k)

# ===== 2) Génération finale =====
# tronquer chaque passage individuellement
MAX_PASSAGE_TOKENS = 1200  # ex : 120 tokens par passage

def truncate_passages(passages):
    truncated = []
    for p in passages:
        truncated.append(truncate_to_tokens(p, max_tokens=MAX_PASSAGE_TOKENS))
    return truncated

def generate_answer(question: str, docs: list, top_k: int = 3) -> str:
    # Limite top_k passages
    docs = docs[:top_k]
    # Tronque chaque passage
    docs = truncate_passages(docs)

    context = "\n".join(f"- {d}" for d in docs)

    prompt = f"""Tu es un assistant spécialisé en énergie.
Question : {question}

Contexte :
{context}

Consignes :
- Réponds uniquement à partir du contexte.
- Si ce n’est pas présent, dis-le clairement.
- Réponse brève (5-7 lignes), en français.
"""

    print("PROMPT: "+prompt)


    # Ici le prompt est sûr d’être sous la limite du modèle
    out = generator(
        prompt,
        max_new_tokens=150,
        num_beams=2,
        do_sample=False
    )
    return out[0]["generated_text"].strip()


# ===== 3) Exemple d’utilisation =====
if __name__ == "__main__":
    question = "Quel est le moyen de production le plus important en france ?"
    results = query_index(question, top_k=3)
    docs = results["documents"][0]

    print("\n🔎 Question :", question)
    print("\n📚 Passages retenus (tronqués) :")
    for d in docs:
        print("-", truncate_to_tokens(d, max_tokens=120))

    print("\n📌 Réponse générée :\n")
    print(generate_answer(question, docs, top_k=3))
