import gradio as gr

import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ✅ On importe tout depuis base_function
from base_function import (
    ask_question_with_BM25_rag,
    ask_question_with_semantic_rag,
    ask_question_with_hybrid_rag,
)

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
# Routeur retrieval
# =========================
def ask_with_selected_retriever(question: str, retriever: str, top_k: int, memory: list | None):
    if retriever == "BM25":
        return ask_question_with_BM25_rag(question, top_k=top_k, memory=memory)
    elif retriever == "Sémantique":
        return ask_question_with_semantic_rag(question, top_k=top_k, memory=memory)
    elif retriever == "Hybride":
        return ask_question_with_hybrid_rag(question, top_k=top_k, memory=memory)
    else:
        # fallback
        return ask_question_with_semantic_rag(question, top_k=top_k, memory=memory)


# =========================
# Fonction de réponse gradio
# =========================
def respond_to_chat(message, history, retriever, top_k):
    memory = history_to_memory(history)

    model_response, answer, updated_memory = ask_with_selected_retriever(
        question=message,
        retriever=retriever,
        top_k=int(top_k),
        memory=memory,
    )

    return answer


# =========================
# UI Gradio
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 💬 RAG Noob Chatbot")

    with gr.Row():
        retriever = gr.Dropdown(
            choices=["BM25", "Sémantique", "Hybride"],
            value="Hybride",
            label="Retriever",
        )
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")

    chat = gr.ChatInterface(
        fn=respond_to_chat,
        additional_inputs=[retriever, top_k],
    )

if __name__ == "__main__":
    demo.launch()