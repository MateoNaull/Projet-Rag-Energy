# src/ingest.py

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import docx
import pypdf
import re


data_path = Path("data/")
all_files = list(data_path.rglob("*.*")) 


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



def chunk_text_by_sentence(text: str, size: int = 500, overlap: int = 50):

    sentences = re.split(r'(?<=[.!?]) +', text)  
    chunks = []
    chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > size:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:] if overlap < len(chunk) else chunk
            word_count = len(chunk)
        chunk.extend(words)
        word_count += len(words)

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks



model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(path="outputs/index")  
collection = client.get_or_create_collection("energie_rag")


for f in all_files:
    print(f"📄 Traitement : {f}")
    text = load_file(f)
    if not text.strip():
        continue

    chunks = chunk_text_by_sentence(text)
    embeddings = [model.encode(c).tolist() for c in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": str(f)}] * len(chunks),
        ids=[f"{f.stem}_{i}" for i in range(len(chunks))]
    )

print("✅ Index créé avec succès et sauvegardé dans outputs/index/")
