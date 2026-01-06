import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()
LLM_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = "rag_noob_collection"


# =========================
# Appel LLM avec mémoire et prompt système
# =========================


def ask_llm_with_memory(
    question: str,
    memory: list | None = None
):
    """
    Envoie une question au LLM avec :
    - un system prompt TOUJOURS présent
    - une mémoire user/assistant
    """
    SYSTEM_PROMPT = """
Réponds courtoisement aux questions sociales et utilises le contexte fourni pour les questions techniques.
"""
    
    if memory is None:
        memory = []

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(memory)
        + [{"role": "user", "content": question}]
    )

    model_response = LLM_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    answer = model_response.choices[0].message.content

    updated_memory = list(memory) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return model_response, answer, updated_memory


# =========================
# Vectorisation de la question
# =========================
def embed_content(chunk: dict, model: str = "text-embedding-3-small") -> list[float]:
    """
    Prend un chunk au format {"content": "...", "metadata": {...}}
    et retourne son embedding (liste de floats).
    """
    text = chunk["content"]

    resp = LLM_client.embeddings.create(
        model=model,
        input=text
    )

    vector = resp.data[0].embedding
    return vector

# =========================
# Recherche vectorielle dans Qdrant (module 2)
# =========================

def search_top_chunks(
    question: str,
    top_k: int = 5,
    query_filter=None,   # optionnel, tu peux laisser None
):
    """
    - vectorise la question via embed_content
    - interroge Qdrant avec query_points
    - retourne les top_k chunks (payload)
    """


    # 2) Embedding de la question (réutilise embed_content)
    question_vector = embed_content({"content": question})

    # 3) Recherche Qdrant (API query_points)
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points

    # 4) Extraction propre
    chunks = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}

        # selon version, le score peut être hit.score ou hit.distance
        score = getattr(hit, "score", None)
        if score is None:
            score = getattr(hit, "distance", None)

        chunks.append({
            "rank": rank,
            "score": score,
            "doc_name": payload.get("doc_name"),
            "chunk_id": payload.get("chunk_id"),
            "content": payload.get("content"),
        })

    return chunks

# =========================
# Formater question compelete pour LLM avec contexte et mémoire (module 1)
# =========================

def ask_question_with_rag(question: str, top_k: int = 5, memory: list | None = None):
    
    
    chunks = search_top_chunks(question, top_k=top_k)
    if not chunks:
        return None, "Je ne sais pas", memory

    context = "\n\n".join(
        f"--- Source ---\nDocument: {c['doc_name']}\nChunk: {c['chunk_id']}\n\n{c['content']}"
        for c in chunks
    )

    print("\n📚 DOCUMENTS RETROUVÉS (RAG) :")
    for c in chunks:
        print(f"- {c['doc_name']} (chunk {c['chunk_id']}, score={c.get('score')})")

    full_question = f"""
=== CONTEXTE ===
{context}

=== QUESTION ===
{question}
""".strip()

    LLM_response = ask_llm_with_memory(full_question, memory)

    
    return LLM_response

# =========================
# Helpers history gradio => memory 
# =========================

def history_to_memory(history):
    """
    Convertit l'historique Gradio (liste de messages riches)
    en memory OpenAI-style : [{"role": "...", "content": "..."}]
    """
    memory = []

    for m in history:
        role = m.get("role")
        content = m.get("content", [])
        text = content[0].get("text") if content and isinstance(content[0], dict) else None
        if role in ("user", "assistant") and text:
            memory.append({"role": role, "content": text})

    return memory

# =========================
# Fonction de réponse gradio
# =========================

def respond_to_chat(message, history):
    print('\nMESSAGE:')
    print(message)
    
    memory = history_to_memory(history)
    print('\nMEMORY:')
    print(memory)
    
    model_response, answer, memory = ask_question_with_rag(message, top_k=5, memory=memory)
    return answer


demo = gr.ChatInterface(respond_to_chat)

if __name__ == "__main__":
    demo.launch()