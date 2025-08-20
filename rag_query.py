# rag_query.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

# ========= 1) Connexion à Chroma =========
client = chromadb.PersistentClient(path="outputs/index")
collection = client.get_collection("energie_rag")

# ========= 2) Embeddings =========
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ========= 3) Init Mistral =========
MISTRAL_API_KEY = os.getenv("mistral-bwNauW2Ir7MF0fh2AAPWYX43yXto0CUP")
llm_client = Mistral(api_key=MISTRAL_API_KEY)

# ========= 4) Query =========
question = input("🔎 Pose ta question : ")

query_embedding = embedder.encode(question).tolist()
results = collection.query(query_embeddings=[query_embedding], n_results=3)

passages = results["documents"][0]
sources = results["metadatas"][0]

# Construire le prompt
context = "\n".join([f"- {p}" for p in passages])
prompt = f"""Tu es un assistant spécialisé en énergie.
Voici des extraits de documents : 
{context}

Question : {question}
Réponds de façon claire et concise en citant les passages utiles.
"""

# ========= 5) Appel à Mistral =========
response = llm_client.chat.complete(
    model="mistral-small",  # ou "mistral-tiny" si tu veux + rapide
    messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message["content"]

# ========= 6) Affichage =========
print("\n📚 Passages utilisés :")
for i, p in enumerate(passages):
    print(f"{i+1}. {p[:200]}...")  # tronqué

print("\n🤖 Réponse générée :\n")
print(answer)
