import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ==============================
# 1. Chargement du modèle d'embedding + index
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_or_create_collection("energie_rag")

# ==============================
# 2. Chargement du LLM Hugging Face
# ==============================
# Ici on prend un modèle de génération léger et gratuit
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)  
# device=-1 = CPU. Si tu as un GPU, remplace par device=0


# ==============================
# 3. Fonction RAG (retrieval + génération)
# ==============================
def rag_answer(question: str, k: int = 3, max_context_tokens: int = 400):
    # Récupération des passages
    query_emb = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)

    passages = results["documents"][0]
    sources = results["metadatas"][0]

    # Tronquer les passages si trop longs
    context = "\n".join(passages)
    context = context[:max_context_tokens]

    # Construire le prompt
    prompt = (
        f"Tu es un expert en énergie. Utilise les passages suivants pour répondre à la question.\n\n"
        f"Contexte :\n{context}\n\n"
        f"Question : {question}\n\n"
        f"Réponse :"
    )

    # Génération
    response = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )[0]["generated_text"]

    # Nettoyage (supprimer le prompt du début)
    answer = response[len(prompt):].strip()

    return answer, list(zip(passages, sources))


# ==============================
# 4. Interface Streamlit
# ==============================
st.title("🔎 RAG Énergie – Demo")
st.write("Posez une question sur le corpus énergétique")

question = st.text_input("Votre question :")

if st.button("Chercher") and question:
    with st.spinner("Recherche en cours..."):
        answer, passages_sources = rag_answer(question)

    st.subheader("📌 Réponse générée :")
    st.write(answer)

    st.subheader("📚 Passages sources :")
    for passage, meta in passages_sources:
        st.markdown(f"- **Source :** {meta['source']}")
        st.write(passage)
