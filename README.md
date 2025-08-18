# RAG Énergie – Assistant IA

Un prototype de Retrieval-Augmented Generation (RAG) spécialisé dans l’énergie.
Le but : poser des questions (ex. consommation, renouvelables, mix énergétique) et obtenir des réponses documentées à partir de rapports ADEME, RTE, etc.

## Structure
- `data/` : documents sources
- `chroma/` : base vectorielle persistante
- `ingest.py` : ingestion et indexation des docs
- `query_topk.py` : test de recherche sémantique

## Installation
```bash
pip install -r requirements.txt
