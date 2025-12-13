from google import genai
import psycopg
from psycopg import Cursor
import os
import time

# --- Dossier contenant tous les fichiers ---
data_folder = "../data/"

# --- Configuration de l'API Gemini ---
client = genai.Client(
    api_key="AIzaSyBFNZ840CTgxh8LokapY8QNnyCYzhUVT4M",
)

# --- Connexion PostgreSQL ---
db_connection_str = "dbname=rag_chatbot user=postgres password=postgres host=localhost port=5432"

# -------------------------------------------------------
# 1) UTILITAIRES
# -------------------------------------------------------

def load_conversation(file_path: str) -> str:
    with open(file_path, "r", encoding="windows-1252", errors="ignore") as file:
        lines = file.read().split("\n")

    cleaned = [
        l.removeprefix("     ").strip()
        for l in lines
        if l.strip() != "" and not l.startswith("<")
    ]

    return "\n".join(cleaned)

def calculate_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values


def embedding_to_vector_str(embedding: list[float]) -> str:
    """Convertit une liste Python → format pgvector."""
    return "[" + ",".join(map(str, embedding)) + "]"


# -------------------------------------------------------
# 2) RECHERCHE VECTORIELLE
# -------------------------------------------------------

def search_similar_conversation(question: str, cur: Cursor):
    query_emb = calculate_embedding(question)
    query_vector = embedding_to_vector_str(query_emb)

    cur.execute("""
        SELECT conversation_id, content
        FROM conversations
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
    """, (query_vector,))

    return cur.fetchone()


# -------------------------------------------------------
# 3) RÉPONSE DU CHATBOT (GEMINI)
# -------------------------------------------------------

def generate_answer(question: str, context: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
        Tu es un assistant intelligent chargé d’analyser une conversation téléphonique.

        Règles :
        - Le contexte est une conversation complète.
        - "c:" = client, "h:" = hôtesse.
        - Réponds en déduisant l'information à partir de la conversation.
        - Ne répète pas la question.
        - Ne reformule pas le contexte.
        - Donne une réponse naturelle et humaine.

        CONVERSATION :
        {context}

        QUESTION :
        {question}

        Réponse :
        """
        )
    return response.text


# -------------------------------------------------------
# 4) INDEXATION DES FICHIERS (À FAIRE UNE SEULE FOIS)
# -------------------------------------------------------

with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:

        cur.execute("DROP TABLE IF EXISTS conversations")
        cur.execute("""
            CREATE TABLE conversations (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT,
                content TEXT,
                embedding VECTOR(768)
            );
        """)

        print("📁 Indexation des conversations...")

        for filename in os.listdir(data_folder):
            if filename.endswith(".txt"):
                path = os.path.join(data_folder, filename)

                conversation_text = load_conversation(path)
                embedding = calculate_embedding(conversation_text)
                vector_str = embedding_to_vector_str(embedding)

                cur.execute("""
                    INSERT INTO conversations (conversation_id, content, embedding)
                    VALUES (%s, %s, %s::vector)
                """, (filename, conversation_text, vector_str))

                time.sleep(0.05)

        print("✅ Indexation terminée")


# -------------------------------------------------------
# 5) CHATBOT RAG INTERACTIF
# -------------------------------------------------------

print("\n🤖 Chatbot RAG prêt ! (exit pour quitter)")

with psycopg.connect(db_connection_str) as conn:
    with conn.cursor() as cur:

        while True:
            question = input("\n🧑‍💻 Vous : ")

            if question.lower() == "exit":
                break

            conv = search_similar_conversation(question, cur)

            if not conv:
                print("🤖 Chatbot : Aucune conversation pertinente trouvée.")
                continue

            conv_id, context = conv

            print(f"\n📚 Conversation trouvée : {conv_id}")

            answer = generate_answer(question, context)

            print("\n🤖 Chatbot :", answer)
