import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Local RAG deps
import numpy as np
import faiss
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# ------------------------- Utilities -------------------------


def load_text(path: str) -> str:
    """Load plain text or extract text from PDF pages."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_sidecar_metadata(path: str) -> Dict[str, Any]:
    """Load metadata from sidecar '<file>.meta.json' if it exists."""
    meta_path = f"{path}.meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def simple_tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer (sufficient for basic BM25)."""
    return [t for t in text.lower().split() if t.strip()]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Character-window chunking with overlap to keep local continuity."""
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]
        chunks.append(chunk.strip())
        if end == n:
            break
        i = max(0, end - overlap)
    return [c for c in chunks if c]


@dataclass
class DocChunk:
    text: str
    source: str
    metadata: Dict[str, Any]


# ------------------------- OpenAI setup -------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generation and embedding models (tunable via env)
GEN_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMB_MODEL = os.getenv("OPENAI_EMB_MODEL",
                      "text-embedding-3-small")  # 1536 dims

# ------------------------- OpenAI File Search (Vector Stores) -------------------------


def openai_create_vector_store(name: str, expires_days: Optional[int]) -> str:
    """Create a vector store (optionally auto-expiring for cost control)."""
    kwargs = {"name": name}
    if expires_days and expires_days > 0:
        kwargs["expires_after"] = {
            "anchor": "last_active_at", "days": expires_days}
    vs = client.vector_stores.create(**kwargs)
    return vs.id


def openai_upload_with_metadata(vector_store_id: str, files: List[str]) -> None:
    """Upload files individually so we can attach per-file metadata if present."""
    for fp in files:
        meta = load_sidecar_metadata(fp)
        with open(fp, "rb") as f:
            client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file={"file_name": os.path.basename(fp), "content": f},
                metadata=meta if meta else None
            )


def openai_rag_with_filter(question: str, vector_store_id: str,
                           metadata_filter: Optional[Tuple[str, str]] = None,
                           k: int = 6) -> str:
    """
    Try to apply a metadata filter; if not available on your tenant/version,
    fall back to a plain file_search call and inform the user.
    """
    tools = [{"type": "file_search", "max_num_results": k}]
    tool_resources = {"file_search": {"vector_store_ids": [vector_store_id]}}
    tool_config: Dict[str, Any] = {}
    if metadata_filter:
        key, val = metadata_filter
        tool_config = {
            "file_search": {
                "ranking_options": {"ranker": "default"},
                "filters": {"where": {key: {"$eq": val}}}
            }
        }
    try:
        resp = client.responses.create(
            model=GEN_MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": question}]}],
            tools=tools,
            tool_resources=tool_resources,
            metadata={"file_search": tool_config} if tool_config else None,
            include=["output_text"]
        )
        return resp.output_text
    except Exception as e:
        resp = client.responses.create(
            model=GEN_MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": question}]}],
            tools=tools,
            tool_resources=tool_resources,
            include=["output_text"]
        )
        return resp.output_text + f"\n\n[warn] metadata filter unavailable for this account/version ({e})."

# ------------------------- Local: BM25 (lexical) -------------------------


class BM25Lex:
    """BM25 over the same local chunks used by the vector index."""

    def __init__(self):
        self.corpus_tokens: List[List[str]] = []
        self.chunks: List[DocChunk] = []
        self.engine: Optional[BM25Okapi] = None

    def add(self, chunks: List[DocChunk]):
        for c in chunks:
            toks = simple_tokenize(c.text)
            self.corpus_tokens.append(toks)
            self.chunks.append(c)
        self.engine = BM25Okapi(self.corpus_tokens)

    def query(self, q: str, top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float]]:
        """Return (global_chunk_index, bm25_score)."""
        if not self.engine or not self.chunks:
            return []
        query_tokens = simple_tokenize(q)
        eligible = list(range(len(self.chunks)))
        if where:
            eligible = [i for i, c in enumerate(self.chunks)
                        if all(c.metadata.get(k) == v for k, v in where.items())]
            if not eligible:
                return []
        scores_all = self.engine.get_scores(query_tokens)
        pairs = [(i, float(scores_all[i])) for i in eligible]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:min(top_k, len(pairs))]

# ------------------------- Local: FAISS (vector) -------------------------


class FaissRAG:
    """FAISS index with cosine similarity (normalized embeddings)."""

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.embs = None  # numpy array [N, dim], normalized
        self.chunks: List[DocChunk] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = client.embeddings.create(model=EMB_MODEL, input=texts).data
        X = np.array([e.embedding for e in embs], dtype="float32")
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def add(self, chunks: List[DocChunk]):
        self.chunks.extend(chunks)
        X = self._embed([c.text for c in chunks])
        self.embs = X if self.embs is None else np.vstack([self.embs, X])
        self.index.add(X)

    def query(self, q: str, top_k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float]]:
        """Return (global_chunk_index, cosine_score)."""
        q_emb = self._embed([q])  # [1, dim]
        eligible = list(range(len(self.chunks)))
        if where:
            eligible = [i for i, c in enumerate(self.chunks)
                        if all(c.metadata.get(k) == v for k, v in where.items())]
            if not eligible:
                return []
        sub = self.embs[eligible, :]  # [M, dim]
        sims = (q_emb @ sub.T).flatten()
        order = np.argsort(-sims)[:min(top_k, len(eligible))]
        return [(eligible[int(i)], float(sims[int(i)])) for i in order]

# ------------------------- Local: Chroma (vector as a service) -------------------------


class ChromaRAG:
    """
    Chroma holds vector search. We keep a local 'catalog' to align doc IDs with local chunks,
    and compute MMR embeddings on the fly for the chosen candidates.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.client = chromadb.PersistentClient(
            path=persist_dir) if persist_dir else chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="local_rag",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMB_MODEL
            )
        )

    def add(self, chunks: List[DocChunk]):
        ids = [f"id-{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **c.metadata} for c in chunks]
        )

    def query(self, q: str, top_k: int = 10, where: Optional[Dict[str, Any]] = None
              ) -> List[Tuple[str, float]]:
        """
        Return (relative_id, approx_score). The caller maps IDs back to local indices.
        """
        res = self.collection.query(
            query_texts=[q], n_results=top_k, where=where or {})
        dists = res.get("distances", [[]])[0] or []
        rel_ids = res.get("ids", [[]])[0]
        scores = [
            1.0 - float(d) if d is not None else 0.0 for d in dists] if dists else [1.0]*len(rel_ids)
        return list(zip(rel_ids, scores))

# ------------------------- Hybrid fusion (vector + BM25) -------------------------


def fuse_hybrid(emb_hits, bm25_hits, top_k: int, alpha: float):
    """
    Weighted late fusion: normalize BM25 and combine with vector scores.
    alpha in [0..1]: 1=all vector, 0=all BM25.
    """
    max_bm = max((s for _, s in bm25_hits), default=1.0) or 1.0
    bm_norm = {i: (s / max_bm) for i, s in bm25_hits}
    emb_norm = {i: s for i, s in emb_hits}
    all_ids = set(emb_norm.keys()) | set(bm_norm.keys())
    fused = []
    for i in all_ids:
        s = alpha * emb_norm.get(i, 0.0) + (1 - alpha) * bm_norm.get(i, 0.0)
        fused.append((i, s))
    fused.sort(key=lambda x: -x[1])
    return fused[:top_k]

# ------------------------- MMR (Maximal Marginal Relevance) -------------------------


def mmr_select(
    query_emb: np.ndarray,           # [dim]
    doc_embs: np.ndarray,            # [N, dim] normalized
    k: int,                          # number to select
    lambda_weight: float = 0.6       # relevance vs. diversity trade-off
) -> List[int]:
    """
    Greedy MMR selection:
    - query_emb dot doc_embs gives cosine similarity (requires normalized vectors).
    - At each step, pick the doc that maximizes:
        lambda * sim(doc, query) - (1 - lambda) * max_sim(doc, already_selected)
    """
    if doc_embs.size == 0 or k <= 0:
        return []
    k = min(k, doc_embs.shape[0])

    # Precompute similarities to query
    sim_to_query = doc_embs @ query_emb  # [N]

    selected: List[int] = []
    candidates = set(range(doc_embs.shape[0]))

    # Select the best by relevance first
    first = int(np.argmax(sim_to_query))
    selected.append(first)
    candidates.remove(first)

    # Iteratively add items balancing relevance and novelty
    while len(selected) < k and candidates:
        # Compute max similarity to any already selected
        sel_embs = doc_embs[selected, :]  # [S, dim]
        # For each candidate, compute max sim to selected set
        sims_to_selected = doc_embs[list(candidates), :] @ sel_embs.T  # [C, S]
        max_sim_to_sel = sims_to_selected.max(
            axis=1) if sims_to_selected.size else np.zeros((len(candidates),))
        cand_list = list(candidates)
        # MMR objective
        mmr_scores = lambda_weight * \
            sim_to_query[cand_list] - (1 - lambda_weight) * max_sim_to_sel
        pick_idx = int(np.argmax(mmr_scores))
        pick = cand_list[pick_idx]
        selected.append(pick)
        candidates.remove(pick)

    return selected

# ------------------------- LLM re-ranking -------------------------


def llm_rerank(question: str, candidates: List[DocChunk], top_m: int = 24) -> List[DocChunk]:
    """
    Ask the LLM to score candidates 0..100 by relevance to the question.
    Returns re-ordered candidates (cut to top_m).
    """
    if not candidates:
        return []
    max_items = min(top_m, len(candidates))
    items = candidates[:max_items]

    def compact(c: DocChunk, i: int) -> Dict[str, Any]:
        txt = c.text.strip().replace("\n", " ")
        if len(txt) > 600:
            txt = txt[:600] + "…"
        return {"i": i, "source": os.path.basename(c.source), "metadata": c.metadata, "text": txt}

    payload = [compact(c, i) for i, c in enumerate(items)]

    instruction = (
        "You are a reranker. Given a QUESTION and N CANDIDATE PASSAGES, "
        "assign each passage an integer relevance score from 0 to 100, "
        "where 100 = extremely relevant and 0 = irrelevant. "
        "Consider explicit factual evidence, exact terms, and synonyms.\n"
        "Respond ONLY with JSON array: "
        '[{\"i\": <index_in_batch>, \"score\": <0..100>}, ...].'
    )

    prompt = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": instruction},
            {"type": "input_text", "text": "QUESTION:\n" + question},
            {"type": "input_text",
                "text": "CANDIDATES (JSON):\n" + json.dumps(payload, ensure_ascii=False)}
        ]
    }

    resp = client.responses.create(
        model=GEN_MODEL,
        input=[prompt],
        temperature=0.0,
        include=["output_text"]
    )

    order = list(range(len(items)))  # fallback if parsing fails
    try:
        scored = json.loads(resp.output_text)
        scored.sort(key=lambda x: -int(x.get("score", 0)))
        order = [int(x["i"]) for x in scored if isinstance(x.get("i"), int)]
        order = [i for i in order if 0 <= i < len(items)]
        if not order:
            order = list(range(len(items)))
    except Exception:
        pass

    return [items[i] for i in order]

# ------------------------- Generation -------------------------


def generate_with_context(question: str, contexts: List[DocChunk], max_ctx_chars: int = 5000) -> str:
    """Concatenate top contexts and ask the LLM to answer strictly based on them."""
    ctx = []
    used = 0
    for c in contexts:
        t = f"[Source: {os.path.basename(c.source)} | meta={c.metadata}] {c.text}\n"
        if used + len(t) > max_ctx_chars:
            break
        ctx.append(t)
        used += len(t)

    prompt = (
        "Answer ONLY using the evidence from the CONTEXT below. "
        "If there is not enough evidence, say you couldn't find it.\n\n"
        "### CONTEXT\n" + "\n".join(ctx) +
        "\n### QUESTION\n" + question +
        "\n### ANSWER:"
    )

    r = client.responses.create(
        model=GEN_MODEL,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt}]}],
        temperature=0.2,
        include=["output_text"]
    )
    return r.output_text

# ------------------------- Ingestion -------------------------


def build_chunks_from_paths(paths: List[str]) -> List[DocChunk]:
    """Load files, build chunks, attach metadata from sidecars."""
    chunks: List[DocChunk] = []
    for p in paths:
        text = load_text(p)
        meta = load_sidecar_metadata(p)
        for ch in chunk_text(text):
            chunks.append(DocChunk(text=ch, source=p, metadata=meta))
    return chunks

# ------------------------- Hybrid Orchestrators (with MMR) -------------------------


class HybridLocalFAISS:
    """
    Fusion (vector + BM25) -> MMR -> optional LLM rerank -> generation.
    Uses stored FAISS embeddings for fast MMR.
    """

    def __init__(self, chunks: List[DocChunk]):
        self.faiss = FaissRAG(dim=1536)
        self.faiss.add(chunks)
        self.bm25 = BM25Lex()
        self.bm25.add(chunks)
        self.chunks = chunks

    def ask(self, q: str, top_k: int, where: Optional[Dict[str, Any]], alpha: float,
            mmr_lambda: float, mmr_keep: int,
            reranker: str = "llm", rerank_top_m: int = 24) -> List[DocChunk]:
        # Retrieve more candidates from each ranker to ensure good recall
        emb_hits = self.faiss.query(q, top_k=max(top_k*4, top_k), where=where)
        bm_hits = self.bm25.query(q, top_k=max(top_k*4, top_k), where=where)

        # Late fusion
        fused = fuse_hybrid(emb_hits, bm_hits, top_k=max(
            top_k*3, top_k), alpha=alpha)
        cand_indices = [i for i, _ in fused]
        cand_chunks = [self.chunks[i] for i in cand_indices]

        # --- MMR over fused candidates (using FAISS-stored embeddings) ---
        # Build query embedding and candidate embedding matrix
        q_emb = self.faiss._embed([q])[0]  # [dim]
        # [N, dim] already normalized
        doc_embs = self.faiss.embs[cand_indices, :]

        # Select a diversified subset (mmr_keep controls how many pass to re-ranker)
        mmr_k = min(max(top_k, 1), mmr_keep, len(cand_indices))
        sel_rel = mmr_select(q_emb, doc_embs, k=mmr_k,
                             lambda_weight=mmr_lambda)
        cand_after_mmr = [cand_chunks[i] for i in sel_rel]

        # Optional LLM re-rank
        if reranker == "llm":
            reranked = llm_rerank(q, cand_after_mmr, top_m=rerank_top_m)
            return reranked[:top_k]
        return cand_after_mmr[:top_k]


class HybridLocalChroma:
    """
    Fusion (vector + BM25) -> MMR -> optional LLM rerank -> generation.
    For MMR, we compute embeddings on-the-fly for the chosen fused candidates.
    """

    def __init__(self, chunks: List[DocChunk], persist_dir: Optional[str] = None):
        self.chunks = chunks
        self.id_map_local = {f"id-{i}": i for i in range(len(chunks))}
        self.chroma = chromadb.PersistentClient(
            path=persist_dir) if persist_dir else chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name="local_rag",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMB_MODEL
            )
        )
        # Keep collection aligned deterministically with local IDs
        if self.collection.count() > 0:
            self.collection.delete(where={})
        self.collection.add(
            ids=[f"id-{i}" for i in range(len(chunks))],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **c.metadata} for c in chunks]
        )
        self.bm25 = BM25Lex()
        self.bm25.add(chunks)

    def _embed_texts_norm(self, texts: List[str]) -> np.ndarray:
        """Create normalized embeddings for arbitrary texts (used for MMR)."""
        embs = client.embeddings.create(model=EMB_MODEL, input=texts).data
        X = np.array([e.embedding for e in embs], dtype="float32")
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def ask(self, q: str, top_k: int, where: Optional[Dict[str, Any]], alpha: float,
            mmr_lambda: float, mmr_keep: int,
            reranker: str = "llm", rerank_top_m: int = 24) -> List[DocChunk]:
        # Vector search via Chroma (ids relative to current collection)
        res = self.collection.query(query_texts=[q], n_results=max(
            top_k*4, top_k), where=where or {})
        rel_ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0] or []
        emb_hits = []
        if rel_ids:
            scores = [
                1.0 - float(d) if d is not None else 0.0 for d in dists] if dists else [1.0]*len(rel_ids)
            for rid, s in zip(rel_ids, scores):
                idx = self.id_map_local.get(rid)
                if idx is not None:
                    emb_hits.append((idx, s))

        # Lexical (BM25)
        bm_hits = self.bm25.query(q, top_k=max(top_k*4, top_k), where=where)

        # Late fusion
        fused = fuse_hybrid(emb_hits, bm_hits, top_k=max(
            top_k*3, top_k), alpha=alpha)
        cand_indices = [i for i, _ in fused]
        cand_chunks = [self.chunks[i] for i in cand_indices]

        # --- MMR over fused candidates (compute embeddings on-the-fly) ---
        q_emb = self._embed_texts_norm([q])[0]  # [dim]
        doc_embs = self._embed_texts_norm(
            [c.text for c in cand_chunks])  # [N, dim]
        mmr_k = min(max(top_k, 1), mmr_keep, len(cand_indices))
        sel_rel = mmr_select(q_emb, doc_embs, k=mmr_k,
                             lambda_weight=mmr_lambda)
        cand_after_mmr = [cand_chunks[i] for i in sel_rel]

        # Optional LLM re-rank
        if reranker == "llm":
            reranked = llm_rerank(q, cand_after_mmr, top_m=rerank_top_m)
            return reranked[:top_k]
        return cand_after_mmr[:top_k]

# ------------------------- CLI -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RAG - OpenAI (metadata filter) + Local HYBRID (FAISS/Chroma + BM25) + MMR + optional LLM Re-Rank")
    parser.add_argument("--backend", choices=["openai", "faiss", "chroma"], required=True,
                        help="Retrieval backend: OpenAI file_search, or local FAISS/Chroma.")
    parser.add_argument("--docs_glob", default="docs/*.*",
                        help="Glob pattern for input files (e.g., 'docs/*.pdf').")
    parser.add_argument("--q", required=True, help="User question.")
    parser.add_argument("--categoria", default=None,
                        help="Metadata filter value for key 'categoria'.")
    parser.add_argument("--expires_days", type=int, default=1,
                        help="OpenAI vector store auto-expire (days).")
    parser.add_argument("--vector_store_name", default="meu_kb",
                        help="OpenAI vector store name.")
    parser.add_argument("--top_k", type=int, default=6,
                        help="Final number of passages to feed the generator.")
    parser.add_argument("--persist_chroma", default=None,
                        help="Chroma persistence directory (optional).")
    parser.add_argument("--hybrid_alpha", type=float, default=0.6,
                        help="Weight for vector score in fusion (0..1).")
    # MMR knobs
    parser.add_argument("--mmr_lambda", type=float, default=0.6,
                        help="MMR lambda (relevance vs. diversity).")
    parser.add_argument("--mmr_keep", type=int, default=24,
                        help="How many passages survive MMR to the next stage.")
    # Re-ranker knobs
    parser.add_argument(
        "--reranker", choices=["none", "llm"], default="llm", help="Apply LLM re-ranking or not.")
    parser.add_argument("--rerank_top_m", type=int, default=24,
                        help="How many candidates enter the LLM reranker.")
    args = parser.parse_args()

    files = [p for p in glob.glob(args.docs_glob)
             if not p.endswith(".meta.json")]
    if not files:
        print("No files found. Adjust --docs_glob.")
        return

    # ----------------- OpenAI File Search -----------------
    if args.backend == "openai":
        vs_id = openai_create_vector_store(
            args.vector_store_name, args.expires_days)
        print(f"[OpenAI] Vector Store: {vs_id}")
        openai_upload_with_metadata(vs_id, files)
        where = ("categoria", args.categoria) if args.categoria else None
        answer = openai_rag_with_filter(args.q, vs_id, where, k=args.top_k)
        print("\n=== ANSWER (OpenAI File Search) ===\n")
        print(answer)
        return

    # ----------------- Local (FAISS / Chroma) -----------------
    
    chunks = build_chunks_from_paths(files)
    where = {"categoria": args.categoria} if args.categoria else None

    if args.backend == "faiss":
        print("[Local] FAISS + BM25 -> MMR -> Re-Rank (optional)")
        engine = HybridLocalFAISS(chunks)
        ctx = engine.ask(
            args.q,
            top_k=args.top_k,
            where=where,
            alpha=args.hybrid_alpha,
            mmr_lambda=args.mmr_lambda,
            mmr_keep=args.mmr_keep,
            reranker=args.reranker,
            rerank_top_m=args.rerank_top_m
        )
        ans = generate_with_context(args.q, ctx)
        print("\n=== ANSWER (Local FAISS + BM25 + MMR + Re-Rank) ===\n")
        print(ans)
        return

    if args.backend == "chroma":
        print("[Local] Chroma + BM25 -> MMR -> Re-Rank (optional)")
        engine = HybridLocalChroma(chunks, persist_dir=args.persist_chroma)
        ctx = engine.ask(
            args.q,
            top_k=args.top_k,
            where=where,
            alpha=args.hybrid_alpha,
            mmr_lambda=args.mmr_lambda,
            mmr_keep=args.mmr_keep,
            reranker=args.reranker,
            rerank_top_m=args.rerank_top_m
        )
        ans = generate_with_context(args.q, ctx)
        print("=== ANSWER (Local Chroma + BM25 + MMR + Re-Rank) ===")
        print(ans)
        return


if __name__ == "__main__":
    main()
