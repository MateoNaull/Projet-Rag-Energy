# app.py
import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ===== 1) Charger la clé API =====
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error("❌ Pas de clé API MISTRAL trouvée. Vérifie ton .env")
    st.stop()

# ===== 2) Initialiser le client Mistral =====
llm_client = Mistral(api_key=api_key)

# ===== 3) Charger l’index Chroma =====
client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_collection("energie_rag")

# Embeddings identiques à ceux d’ingest.py
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===== 4) Fonction RAG =====
def rag_query(question: str, k: int = 3):
    # Encoder la question
    q_embedding = embedder.encode(question).tolist()

    # Récupérer les meilleurs passages
    results = collection.query(query_embeddings=[q_embedding], n_results=k)
    documents = results["documents"][0]
    sources = results["metadatas"][0]

    # Contexte
    context = "\n\n".join(documents)

    # Prompt
    prompt = f"""
    Tu es un assistant expert en énergie.
    Utilise uniquement les passages suivants pour répondre à la question de manière claire et concise.

    Contexte :
    {context}

    Question :
    {question}

    Réponse :
    """

    # Appel au modèle
    resp = llm_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content, documents, sources

# ===== 5) Interface Streamlit =====
st.title("⚡ RAG Énergie - Assistant IA")
st.write("Pose une question et je vais chercher dans mon corpus documentaire.")

question = st.text_input("💬 Ta question :", "")

if st.button("🔎 Interroger") and question.strip():
    with st.spinner("Recherche et génération en cours..."):
        answer, docs, sources = rag_query(question)

    st.subheader("📌 Réponse")
    st.write(answer)

    st.subheader("📚 Passages utilisés")
    for i, doc in enumerate(docs):
        st.markdown(f"**Source {i+1}** ({sources[i].get('source', 'inconnu')}) :")
        st.write(doc[:400] + "...")
