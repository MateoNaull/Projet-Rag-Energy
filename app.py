import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error("❌ Pas de clé API MISTRAL trouvée. Vérifie ton .env")
    st.stop()

llm_client = Mistral(api_key=api_key)

client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_collection("energie_rag")

embedder = SentenceTransformer("all-MiniLM-L6-v2")   #Le même embedder que dans ingest.py


def rag_query(question: str, k: int = 3):
    q_embedding = embedder.encode(question).tolist()

    results = collection.query(query_embeddings=[q_embedding], n_results=k)
    documents = results["documents"][0]
    sources = results["metadatas"][0]

    context = "\n\n".join(documents)

    prompt = f"""
    Tu es un assistant expert en énergie.
    Utilise uniquement les passages suivants pour répondre à la question de manière claire et concise.

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

    return resp.choices[0].message.content, documents, sources




#===== Interface Streamlit =====

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
