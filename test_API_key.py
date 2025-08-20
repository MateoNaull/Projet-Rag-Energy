# test_mistral_key.py
import os
from mistralai import Mistral
from dotenv import load_dotenv
import os

# Charger le fichier .env
load_dotenv()

# Récupérer la clé
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("❌ Pas de clé API MISTRAL trouvée. Vérifie ton fichier .env")

print("✅ Clé API chargée :", api_key[:6] + "..." )  # Affiche juste le début pour vérifier



api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("❌ Aucune clé détectée dans MISTRAL_API_KEY")
    exit(1)

client = Mistral(api_key=api_key)

try:
    resp = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Bonjour, ceci est un test"}]
    )
    print("✅ Clé API valide ! Réponse reçue :", resp.choices[0].message.content)

except Exception as e:
    print("❌ Erreur détectée :", e)

