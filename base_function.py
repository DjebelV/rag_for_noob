import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

load_dotenv()


# ========================= FONCTIONS RECHERCHE SEMANTIQUE =========================


LLM_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = "rag_noob_collection"

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
Réponds courtoisement aux questions sociales, ou utilises le contexte fourni pour les questions techniques. 
Tu dois répondre uniquement à partir du contexte fournis. 
Si l'information n'est pas présente dans le contexte, réponds explicitement : "Je ne sais pas".
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


def search_top_chunks_semantique(
    question: str,
    top_k: int = 5,
    query_filter=None,   # optionnel, tu peux laisser None
):
    """
    - vectorise la question via embed_content
    - interroge Qdrant avec query_points
    - retourne les top_k chunks (payload)
    """

    # 1) Client Qdrant
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

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


def ask_question_with_semantic_rag(
    question: str,
    top_k: int = 5,
    memory: list | None = None,
):
    """
    Pose une question en mode RAG :
    - recherche les top_k chunks via search_top_chunks
    - construit un contexte à partir de ces chunks
    - interroge le LLM via ask_llm_with_memory
    - retourne (model_response, answer, updated_memory)
    """

    # 1) Recherche des chunks pertinents
    chunks = search_top_chunks_semantique(question, top_k=top_k)

    if not chunks:
        raise ValueError("Aucun chunk trouvé pour la question")

    # 2) Construction du contexte
    context_blocks = []
    for c in chunks:
        block = f"""
--- Source ---
Document: {c['doc_name']}
Chunk: {c['chunk_id']}

{c['content']}
""".strip()
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    # 3) Prompt final
    full_question = f"""

=== CONTEXTE ===
{context}

=== QUESTION ===
{question}
""".strip()

    # 4) Appel LLM
    model_response, answer, updated_memory = ask_llm_with_memory(
        full_question,
        memory
    )

    return model_response, answer, updated_memory

# ========================= FONCTIONS RECHERCHE BM25 =========================

from pathlib import Path
import json
import re
from collections import Counter
import math
from typing import List, Dict, Any, Optional

# Tokenisation simple (robuste pour FR + chiffres)
def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-zàâçéèêëîïôùûüÿñæœ0-9]+", text)

# 1) Build index BM25 global
def build_bm25_global_index(
    chunks_dir: Path = Path("../data/chunks"),
    out_path: Path = Path("../data/bm25/bm25_global.json"),
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, Any]:
    """
    Construit un index BM25 GLOBAL sur tous les chunks présents dans ../data/chunks/*.json
    et sauvegarde un fichier unique: ../data/bm25/bm25_global.json

    Le fichier contient :
    - paramètres BM25
    - stats corpus (N, avgdl)
    - chunks (doc_name, chunk_id, content, tokens_len)
    - idf global
    - tf par chunk (Counter -> dict)
    - doc_len par chunk

    Retourne le dict d'index (JSON-serializable).
    """
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {chunks_dir}")

    chunk_files = sorted(chunks_dir.glob("*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"Aucun fichier .json dans : {chunks_dir}")

    print(f"🚀 Build BM25 GLOBAL sur {len(chunk_files)} fichiers chunks")

    all_chunks: List[Dict[str, Any]] = []
    tokenized_chunks: List[List[str]] = []
    doc_lens: List[int] = []
    df = Counter()  # document frequency = nb de chunks contenant le terme

    # 1) Charger + tokeniser tous les chunks
    total_chunks = 0
    for i, path in enumerate(chunk_files, start=1):
        doc_name = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            raise ValueError(f"Fichier chunks invalide (pas une liste) : {path}")

        print(f"📄 [{i}/{len(chunk_files)}] {doc_name} → {len(data)} chunks")
        for j, c in enumerate(data):
            content = c.get("content", "") or ""
            chunk_id = c.get("chunk_id", j)

            toks = tokenize(content)
            tokenized_chunks.append(toks)
            doc_lens.append(len(toks))

            # df: compte 1 fois par chunk
            for t in set(toks):
                df[t] += 1

            all_chunks.append({
                "doc_name": doc_name,
                "chunk_id": chunk_id,
                "content": content,
                "tokens_len": len(toks),
            })
            total_chunks += 1

    if total_chunks == 0:
        raise ValueError("Aucun chunk chargé (corpus vide)")

    N = total_chunks
    avgdl = sum(doc_lens) / max(1, N)

    # 2) Calcul IDF global
    idf = {}
    for t, dft in df.items():
        # IDF BM25 standard (avec +1.0 pour éviter valeurs négatives)
        idf[t] = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)

    # 3) Calcul TF par chunk
    tf = [dict(Counter(toks)) for toks in tokenized_chunks]

    index = {
        "type": "bm25_global",
        "params": {"k1": k1, "b": b},
        "stats": {
            "N_chunks": N,
            "avgdl": avgdl,
            "N_terms": len(idf),
            "source_dir": str(chunks_dir),
        },
        "chunks": all_chunks,
        "idf": idf,
        "tf": tf,
        "doc_len": doc_lens,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Index BM25 global écrit → {out_path}")
    print(f"   - chunks: {N}")
    print(f"   - termes: {len(idf)}")
    print(f"   - avgdl : {avgdl:.2f}")


# %%

def load_bm25_global_index(
    path: Path = Path("../data/bm25/bm25_global.json")
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index BM25 introuvable : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def bm25_score_one_chunk(
    q_tokens: List[str],
    tf_chunk: Dict[str, int],
    doc_len: int,
    idf: Dict[str, float],
    avgdl: float,
    k1: float,
    b: float,
) -> float:
    """
    Score BM25 d'un chunk pour une requête tokenisée.
    """
    score = 0.0
    denom_const = k1 * (1.0 - b + b * (doc_len / (avgdl if avgdl > 0 else 1.0)))

    for t in q_tokens:
        if t not in tf_chunk:
            continue
        tf = tf_chunk[t]
        # terme absent de l'idf (rare) => 0
        idf_t = idf.get(t, 0.0)
        score += idf_t * (tf * (k1 + 1.0)) / (tf + denom_const)

    return float(score)

def search_top_chunks_bm25(
    question: str,
    top_k: int = 5,
    index: Optional[Dict[str, Any]] = None,
    index_path: Path = Path("../data/bm25/bm25_global.json"),
) -> List[Dict[str, Any]]:
    """
    Retourne les top_k chunks selon BM25 global.
    Format :
    [{rank, score, doc_name, chunk_id, content}]
    """
    if index is None:
        index = load_bm25_global_index(index_path)

    q_tokens = tokenize(question)
    k1 = index["params"]["k1"]
    b = index["params"]["b"]
    avgdl = index["stats"]["avgdl"]
    idf = index["idf"]

    scores = []
    for i in range(index["stats"]["N_chunks"]):
        s = bm25_score_one_chunk(
            q_tokens=q_tokens,
            tf_chunk=index["tf"][i],
            doc_len=index["doc_len"][i],
            idf=idf,
            avgdl=avgdl,
            k1=k1,
            b=b,
        )
        scores.append((s, i))

    # top-k
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_k]

    results = []
    for rank, (s, i) in enumerate(top, start=1):
        c = index["chunks"][i]
        results.append({
            "rank": rank,
            "score": s,
            "doc_name": c["doc_name"],
            "chunk_id": c["chunk_id"],
            "content": c["content"],
        })
    return results


def ask_question_with_BM25_rag(
    question: str,
    top_k: int = 5,
    memory: list | None = None,
):
    """
    Pose une question en mode RAG :
    - recherche les top_k chunks via search_top_chunks
    - construit un contexte à partir de ces chunks
    - interroge le LLM via ask_llm_with_memory
    - retourne (model_response, answer, updated_memory)
    """

    # 1) Recherche des chunks pertinents
    chunks = search_top_chunks_bm25(question, top_k=top_k)

    if not chunks:
        raise ValueError("Aucun chunk trouvé pour la question")

    # 2) Construction du contexte
    context_blocks = []
    for c in chunks:
        block = f"""
--- Source ---
Document: {c['doc_name']}
Chunk: {c['chunk_id']}

{c['content']}
""".strip()
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    # 3) Prompt final
    full_question = f"""

=== CONTEXTE ===
{context}

=== QUESTION ===
{question}
""".strip()

    # 4) Appel LLM
    model_response, answer, updated_memory = ask_llm_with_memory(
        full_question,
        memory
    )

    return model_response, answer, updated_memory


# ========================= FONCTIONS RECHERCHE HYBRIDE =========================

COLLECTION_NAME_HYBRIDE = "rag_noob_collection_hybride"


def search_top_chunks_hybrid(
    question: str,
    top_k: int = 5,
    query_filter=None,
):
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,  # nécessaire si tu utilises Document(model="Qdrant/bm25")
    )

    question_vector = embed_content({"content": question})

    res = qdrant_client.query_points(
        collection_name=COLLECTION_NAME_HYBRIDE,
        prefetch=[
            # 1) Dense
            qdrant_models.Prefetch(
                query=question_vector,
                using="dense",
                limit=top_k,
                filter=query_filter,
            ),
            # 2) BM25 natif
            qdrant_models.Prefetch(
                query=qdrant_models.Document(
                    text=question,
                    model="Qdrant/bm25",
                ),
                using="bm25",
                limit=top_k,
                filter=query_filter,
            ),
        ],
        # fusion côté serveur 
        query=qdrant_models.FusionQuery(
            fusion=qdrant_models.Fusion.RRF
        ),
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for rank, hit in enumerate(res.points, start=1):
        payload = hit.payload or {}
        chunks.append({
            "rank": rank,
            "score": getattr(hit, "score", None),
            "doc_name": payload.get("doc_name"),
            "chunk_id": payload.get("chunk_id"),
            "content": payload.get("content"),
        })
    return chunks


def ask_question_with_hybrid_rag(
    question: str,
    top_k: int = 5,
    memory: list | None = None,
):
    """
    Pose une question en mode RAG :
    - recherche les top_k chunks via search_top_chunks
    - construit un contexte à partir de ces chunks
    - interroge le LLM via ask_llm_with_memory
    - retourne (model_response, answer, updated_memory)
    """

    # 1) Recherche des chunks pertinents
    chunks = search_top_chunks_hybrid(question, top_k=top_k)

    if not chunks:
        raise ValueError("Aucun chunk trouvé pour la question")

    # 2) Construction du contexte
    context_blocks = []
    for c in chunks:
        block = f"""
--- Source ---
Document: {c['doc_name']}
Chunk: {c['chunk_id']}

{c['content']}
""".strip()
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    # 3) Prompt final
    full_question = f"""

=== CONTEXTE ===
{context}

=== QUESTION ===
{question}
""".strip()

    # 4) Appel LLM
    model_response, answer, updated_memory = ask_llm_with_memory(
        full_question,
        memory
    )

    return model_response, answer, updated_memory
