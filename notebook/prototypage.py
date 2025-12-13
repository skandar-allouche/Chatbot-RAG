import google.generativeai as genai
import psycopg
from psycopg import Cursor
import os
import time

# --- Dossier contenant tous les fichiers ---
data_folder = "../data/"

# --- Configuration de l'API Gemini ---
genai.configure(api_key="AIzaSyCU51GOJaUR-lUEF-IWcxr5t9dE65VuHN4")

# --- Connexion PostgreSQL ---
db_connection_str = "dbname=rag_chatbot user=postgres password=postgres host=localhost port=5432"

# -------------------------------------------------------
# 1) UTILITAIRES
# -------------------------------------------------------

def load_file(file_path: str) -> list[str]:
    """Charge un fichier et extrait les lignes utiles."""
    with open(file_path, "r", encoding="windows-1252", errors="ignore") as file:
        lines = file.read().split("\n")

    cleaned = [
        l.removeprefix("     ").strip()
        for l in lines
        if l.strip() != "" and not l.startswith("<")
    ]
    return cleaned


def calculate_embedding(text: str) -> list[float]:
    """Génère un embedding avec Gemini."""
    response = genai.embed_content(
        model="text-embedding-004",
        content=text
    )
    return response["embedding"]


def embedding_to_vector_str(embedding: list[float]) -> str:
    """Convertit une liste Python → format pgvector."""
    return "[" + ",".join(map(str, embedding)) + "]"


# -------------------------------------------------------
# 2) RECHERCHE VECTORIELLE
# -------------------------------------------------------

def search_similar(text: str, cur: Cursor, k: int = 5):
    """Recherche les passages les plus proches via pgvector."""
    query_emb = calculate_embedding(text)
    query_vector = embedding_to_vector_str(query_emb)

    cur.execute("""
        SELECT corpus, embedding <=> %s::vector AS distance
        FROM embeddings
        ORDER BY distance ASC
        LIMIT %s;
    """, (query_vector, k))

    return cur.fetchall()


# -------------------------------------------------------
# 3) RÉPONSE DU CHATBOT (GEMINI)
# -------------------------------------------------------

def generate_answer(question: str, context: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        f"""
        Tu es un chatbot intelligent. Voici le contexte trouvé :

        CONTEXTE :
        {context}

        QUESTION :
        {question}

        Réponds de manière claire, précise, et basée uniquement sur le contexte.
        """
    )

    return response.text


# -------------------------------------------------------
# 4) INDEXATION DES FICHIERS (À FAIRE UNE SEULE FOIS)
# -------------------------------------------------------

with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:

        print("🔄 Réinitialisation de la table...")

        cur.execute("DROP TABLE IF EXISTS embeddings")
        cur.execute("""
            CREATE TABLE embeddings (
                id SERIAL PRIMARY KEY,
                corpus TEXT,
                embedding VECTOR(768)
            );
        """)

        print("\n📁 Parcours du dossier :", data_folder)

        for filename in os.listdir(data_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_folder, filename)
                print(f"\n📄 Processing: {file_path}")

                lines = load_file(file_path)
                print(f" → {len(lines)} lignes détectées")

                for line in lines:
                    embedding = calculate_embedding(line)
                    vector_str = embedding_to_vector_str(embedding)

                    cur.execute(
                        """INSERT INTO embeddings (corpus, embedding)
                           VALUES (%s, %s::vector)""",
                        (line, vector_str)
                    )

                    time.sleep(0.05)

        print("\n🎉 Tous les fichiers ont été entièrement indexés !")


# -------------------------------------------------------
# 5) CHATBOT RAG INTERACTIF
# -------------------------------------------------------

print("\n🤖 Chatbot RAG prêt ! Pose une question (or 'exit'): ")

with psycopg.connect(db_connection_str) as conn:
    with conn.cursor() as cur:

        while True:
            question = input("\n🧑‍💻 Vous : ")

            if question.lower() == "exit":
                print("👋 Fin du chatbot.")
                break

            print("🔍 Recherche dans la base...")

            results = search_similar(question, cur, k=5)
            context = "\n".join([row[0] for row in results])

            print("\n📚 Contexte récupéré :")
            print(context)

            answer = generate_answer(question, context)

            print("\n🤖 Chatbot :", answer)
