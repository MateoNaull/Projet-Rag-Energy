# src/ingest.py

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import docx
import pypdf

# ========= 1) Récupération des fichiers =========
data_path = Path("data/")
all_files = list(data_path.rglob("*.*"))  # prend tous les fichiers de data/


# ========= 2) Extraction du texte =========
def load_file(path: Path) -> str:
    """Retourne le texte brut d'un fichier selon son type"""
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text

    elif path.suffix.lower() == ".docx":
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        text = df.astype(str).apply(" ".join, axis=1).str.cat(sep="\n")
        return text

    else:
        print(f"⚠️ Format non supporté : {path}")
        return ""


# ========= 3) Découpage en chunks =========
def chunk_text(text: str, size: int = 500, overlap: int = 50):
    """Découpe un texte en morceaux (chunks) pour l'indexation"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


# ========= 4) Initialisation des embeddings =========
model = SentenceTransformer("all-MiniLM-L6-v2")


# ========= 5) Connexion à Chroma =========
client = chromadb.PersistentClient(path="outputs/index")  # sauvegarde l’index
collection = client.get_or_create_collection("energie_rag")


# ========= 6) Pipeline complet =========
for f in all_files:
    print(f"📄 Traitement : {f}")
    text = load_file(f)
    if not text.strip():
        continue

    chunks = chunk_text(text)
    embeddings = [model.encode(c).tolist() for c in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": str(f)}] * len(chunks),
        ids=[f"{f.stem}_{i}" for i in range(len(chunks))]
    )

print("✅ Index créé avec succès et sauvegardé dans outputs/index/")
