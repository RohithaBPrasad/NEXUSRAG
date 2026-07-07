from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .loaders import describe_image
from .memory import Memory
from .store import Chunk, Store, cite_sources, format_context
from .utils import llm, safe_ollama_embed

# =================================================================
# DYNAMIC SYSTEM PROMPT
# =================================================================

_CODE_QUESTION_WORDS = (
    "fix", "bug", "error", "exception", "traceback", "debug", "solution",
    "broken", "crash", "fails", "failing", "not working", "doesn't work",
    "wrong", "issue", "refactor", "optimize", "optimise",
)


def is_code_question(query: str) -> bool:
    ql = query.lower()
    return any(w in ql for w in _CODE_QUESTION_WORDS)


def chunks_contain_code(chunks: List[Chunk]) -> bool:
    return any(c.doc_type == "code" for c in chunks)


def build_system_prompt(chunks: Optional[List[Chunk]] = None, detailed: bool = False) -> str:
    chunks = chunks or []
    code_mode = chunks_contain_code(chunks)

    if code_mode:
        base = (
            "You are an expert software engineer reviewing the provided source code "
            "context to help solve a real problem. Identify the specific bug, error, "
            "or issue. Explain the root cause, then provide a CONCRETE corrected code "
            "block (not just a description) that fixes it. Reference exact source "
            "filenames/sources from the numbered context when relevant."
        )
    else:
        base = (
            "You are a helpful assistant that answers questions using ONLY the provided "
            "context. If the context does not contain the answer, say so clearly instead "
            "of guessing. Cite which numbered source(s) you used."
        )

    if detailed:
        base += (
            " Give a DETAILED, long-form answer: explain your reasoning step by step, "
            "cover edge cases or caveats where relevant, and use examples to illustrate "
            "points. Do not artificially shorten the answer — thoroughness is preferred "
            "over brevity for this request."
        )

    return base


SIMPLE_RAG_SYSTEM = build_system_prompt()


# =================================================================
# STRATEGY 12 — NAIVE RAG
# =================================================================

def mode_naive_rag(store: Store, query: str, memory: Memory, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    chunks = store.vector_search(query, k=1)
    context = chunks[0].text if chunks else "(no context found)"
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    if detailed:
        prompt += " (Give a detailed, long-form answer with reasoning.)"
    answer = llm(prompt)
    return {"answer": answer, "chunks_used": chunks, "notes": "top-1 chunk only, no reranking"}


# =================================================================
# STRATEGY 1 — SIMPLE RAG
# =================================================================

def mode_simple_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    chunks = store.vector_search(query, k=k)
    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(chunks, detailed))
    return {"answer": answer, "chunks_used": chunks}


# =================================================================
# STRATEGY 2 — SIMPLE RAG WITH MEMORY
# =================================================================

def mode_simple_rag_memory(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    chunks = store.vector_search(query, k=k)
    context = format_context(chunks)
    prompt = (
        f"Conversation so far:\n{memory.as_text()}\n\n"
        f"Context for the current question:\n{context}\n\n"
        f"Current question: {query}"
    )
    system = build_system_prompt(chunks, detailed) + " Use the conversation history to resolve pronouns or follow-ups."
    answer = llm(prompt, system=system)
    return {"answer": answer, "chunks_used": chunks}


# =================================================================
# STRATEGY 13 — ADVANCED RAG
# =================================================================

def _rewrite_query(query: str, memory: Memory) -> str:
    prompt = (
        f"Conversation history:\n{memory.as_text()}\n\n"
        f"Original user query: \"{query}\"\n\n"
        "Rewrite this as a single, fully self-contained search query optimised for "
        "retrieval (resolve pronouns, expand abbreviations if obvious). "
        "Reply with ONLY the rewritten query, nothing else."
    )
    rewritten = llm(prompt)
    return rewritten.strip().strip('"')


def _llm_rerank(query: str, chunks: List[Chunk], top_n: int) -> List[Chunk]:
    if len(chunks) <= top_n:
        return chunks
    listing = "\n".join(f"{i}. {c.text[:300]}" for i, c in enumerate(chunks))
    prompt = (
        f"Query: {query}\n\nCandidate passages:\n{listing}\n\n"
        f"Return the indices of the {top_n} MOST relevant passages to the query, "
        f"most relevant first, as a comma-separated list of numbers only (e.g. 2,0,5)."
    )
    raw = llm(prompt)
    indices = [int(x) for x in re.findall(r"\d+", raw)][:top_n]
    valid = [i for i in indices if 0 <= i < len(chunks)]
    if not valid:
        return chunks[:top_n]
    return [chunks[i] for i in valid]


def mode_advanced_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    rewritten = _rewrite_query(query, memory) if memory.turns else query
    candidates = store.hybrid_search(rewritten, k=k * 3)
    reranked = _llm_rerank(rewritten, candidates, top_n=k)
    context = format_context(reranked)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(reranked, detailed))
    return {
        "answer": answer, "chunks_used": reranked,
        "notes": f"rewritten query: \"{rewritten}\"",
    }


# =================================================================
# STRATEGY 14 — HyDE
# =================================================================

def mode_hyde(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    hypothetical = llm(
        f"Write a short, plausible passage (3-5 sentences) that would answer this "
        f"question, as if it came directly from a reference document. "
        f"Do not say you're unsure — just write the hypothetical passage.\n\nQuestion: {query}"
    )
    hyde_embedding = safe_ollama_embed(hypothetical)
    chunks = store.vector_search(query, k=k, query_embedding=hyde_embedding)
    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(chunks, detailed))
    return {"answer": answer, "chunks_used": chunks, "notes": f"hypothetical doc: \"{hypothetical[:150]}...\""}


# =================================================================
# STRATEGY 6 — BRANCHED RAG
# =================================================================

def _decompose_query(query: str) -> List[str]:
    prompt = (
        f"Question: \"{query}\"\n\n"
        "If this question has multiple independent parts (e.g. a comparison, "
        "a list of causes AND effects, multiple sub-topics), break it into "
        "2-4 standalone sub-questions, one per line, no numbering. "
        "If it is already a single simple question, reply with just that one question."
    )
    raw = llm(prompt)
    sub_qs = [line.strip("-•* ").strip() for line in raw.splitlines() if line.strip()]
    return sub_qs[:4] if sub_qs else [query]


def mode_branched_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    sub_questions = _decompose_query(query)
    branches = []
    all_chunks: List[Chunk] = []
    for sq in sub_questions:
        chunks = store.vector_search(sq, k=max(2, k // max(1, len(sub_questions))))
        branches.append((sq, chunks))
        all_chunks.extend(chunks)
    branch_text = "\n\n".join(
        f"Sub-question: {sq}\nRetrieved context:\n{format_context(chunks)}"
        for sq, chunks in branches
    )
    prompt = (
        f"Original question: {query}\n\n"
        f"Research was done in branches:\n{branch_text}\n\n"
        "Synthesise ONE coherent final answer to the original question using "
        "the branch findings above."
    )
    answer = llm(prompt, system=build_system_prompt(all_chunks, detailed))
    return {
        "answer": answer, "chunks_used": all_chunks,
        "notes": f"sub-questions: {sub_questions}",
    }


# =================================================================
# STRATEGY 4 — GRAPH RAG
# =================================================================

def mode_graph_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    chunks = store.graph_search(query, k=k)
    context = format_context(chunks)
    prompt = f"Context (retrieved via entity-graph traversal):\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(chunks, detailed))
    return {"answer": answer, "chunks_used": chunks, "notes": "retrieved via entity graph traversal"}


# =================================================================
# STRATEGY 9 — SPECULATIVE RAG
# =================================================================

def mode_speculative_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    draft_chunks = store.vector_search(query, k=1)
    draft_prompt = f"Context:\n{format_context(draft_chunks)}\n\nQuestion: {query}\nGive a brief draft answer."
    draft_answer = llm(draft_prompt)

    verify_chunks = store.vector_search(query, k=k * 2)
    verify_prompt = (
        f"Draft answer to verify: \"{draft_answer}\"\n\n"
        f"Question: {query}\n\n"
        f"Full context:\n{format_context(verify_chunks)}\n\n"
        "Check the draft answer against this fuller context. If it is correct and "
        "complete, restate it cleanly. If it is wrong, incomplete, or unsupported "
        "by the context, correct it. Output only the final, verified answer."
    )
    final_answer = llm(verify_prompt, system=build_system_prompt(verify_chunks, detailed))
    return {
        "answer": final_answer, "chunks_used": verify_chunks,
        "notes": f"draft was: \"{draft_answer[:150]}...\"",
    }


# =================================================================
# STRATEGY 10 — CORRECTIVE RAG (CRAG)
# =================================================================

def _grade_relevance(query: str, chunk: Chunk) -> bool:
    prompt = (
        f"Question: {query}\nPassage: {chunk.text[:500]}\n\n"
        "Is this passage relevant and useful for answering the question? "
        "Reply with exactly one word: YES or NO."
    )
    verdict = llm(prompt).strip().upper()
    return verdict.startswith("Y")


def mode_corrective_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    chunks = store.vector_search(query, k=k)
    graded = [(c, _grade_relevance(query, c)) for c in chunks]
    relevant = [c for c, ok in graded if ok]

    notes = f"{len(relevant)}/{len(chunks)} chunks graded relevant"
    if len(relevant) < max(1, len(chunks) // 2):
        rewritten = _rewrite_query(query, memory)
        retry_chunks = store.hybrid_search(rewritten, k=k)
        retry_graded = [(c, _grade_relevance(query, c)) for c in retry_chunks]
        retry_relevant = [c for c, ok in retry_graded if ok]
        notes += f"; low relevance triggered retry with rewritten query \"{rewritten}\" -> {len(retry_relevant)} relevant"
        relevant = retry_relevant or relevant

    if not relevant:
        return {
            "answer": "I don't have enough relevant information in the knowledge base to answer this confidently.",
            "chunks_used": [], "notes": notes,
        }

    context = format_context(relevant)
    prompt = f"Context (pre-filtered for relevance):\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(relevant, detailed))
    return {"answer": answer, "chunks_used": relevant, "notes": notes}


# =================================================================
# STRATEGY 5 — SELF-RAG
# =================================================================

def _needs_retrieval(query: str) -> bool:
    prompt = (
        f"Question: \"{query}\"\n\n"
        "Does answering this well require looking up specific facts from a "
        "knowledge base (vs. something answerable from general reasoning/greetings/opinions)? "
        "Reply with exactly one word: YES or NO."
    )
    return llm(prompt).strip().upper().startswith("Y")


def _self_critique_grounded(query: str, answer: str, context: str) -> bool:
    prompt = (
        f"Context:\n{context}\n\nQuestion: {query}\nProposed answer: {answer}\n\n"
        "Is the proposed answer FULLY supported by the context (no invented facts)? "
        "Reply with exactly one word: YES or NO."
    )
    return llm(prompt).strip().upper().startswith("Y")


def mode_self_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    if not _needs_retrieval(query):
        system = "Answer directly and concisely." if not detailed else "Answer directly, but in detail, with reasoning and examples."
        answer = llm(query, system=system)
        return {"answer": answer, "chunks_used": [], "notes": "self-decided: no retrieval needed"}

    chunks = store.vector_search(query, k=k)
    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(chunks, detailed))

    grounded = _self_critique_grounded(query, answer, context)
    notes = "self-critique: grounded on first pass"
    if not grounded:
        wider_chunks = store.vector_search(query, k=k * 2)
        wider_context = format_context(wider_chunks)
        retry_prompt = (
            f"Context:\n{wider_context}\n\nQuestion: {query}\n\n"
            "Your previous attempt may have included unsupported claims. "
            "Answer again using ONLY what's explicitly stated in this context, "
            "and say what's missing if the context is incomplete."
        )
        answer = llm(retry_prompt, system=build_system_prompt(wider_chunks, detailed))
        chunks = wider_chunks
        notes = "self-critique: first pass ungrounded, retried with wider context"

    return {"answer": answer, "chunks_used": chunks, "notes": notes}


# =================================================================
# STRATEGY 11 — MODULAR RAG
# =================================================================

RETRIEVER_MODULES: Dict[str, Callable[[Store, str, int], List[Chunk]]] = {
    "vector": lambda store, q, k: store.vector_search(q, k=k),
    "keyword": lambda store, q, k: store.keyword_search(q, k=k),
    "hybrid": lambda store, q, k: store.hybrid_search(q, k=k),
    "graph": lambda store, q, k: store.graph_search(q, k=k),
}

RERANKER_MODULES: Dict[str, Callable[[str, List[Chunk], int], List[Chunk]]] = {
    "none": lambda q, chunks, k: chunks[:k],
    "llm": _llm_rerank,
}


def mode_modular_rag(
    store: Store, query: str, memory: Memory, k: int = None,
    retriever: str = "hybrid", reranker: str = "llm", detailed: bool = False, **kwargs,
) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    retrieve_fn = RETRIEVER_MODULES.get(retriever, RETRIEVER_MODULES["hybrid"])
    rerank_fn = RERANKER_MODULES.get(reranker, RERANKER_MODULES["none"])

    candidates = retrieve_fn(store, query, k * 2)
    final_chunks = rerank_fn(query, candidates, k)
    context = format_context(final_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, system=build_system_prompt(final_chunks, detailed))
    return {
        "answer": answer, "chunks_used": final_chunks,
        "notes": f"modules used: retriever={retriever}, reranker={reranker}",
    }


# =================================================================
# STRATEGY 3 — AGENTIC RAG
# =================================================================

AGENTIC_MAX_STEPS = 4

AGENTIC_SYSTEM = (
    "You are an agent that answers questions by searching a knowledge base. "
    "At each step, respond with EXACTLY ONE of:\n"
    "  SEARCH: <a specific search query>\n"
    "  ANSWER: <your final answer>\n"
    "Use SEARCH when you need more/different information than what you've seen. "
    "Use ANSWER only once you have enough information to answer well. "
    "Do not search more than necessary."
)


def mode_agentic_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    transcript = [f"Question: {query}"]
    all_chunks: List[Chunk] = []
    actions_log = []

    def _expand_if_detailed(short_answer: str) -> str:
        if not detailed:
            return short_answer
        return llm(
            f"Expand this answer into a detailed, long-form response with full "
            f"reasoning, using the gathered context below where helpful.\n\n"
            f"Original question: {query}\nShort answer: {short_answer}\n\n"
            f"Gathered context:\n{format_context(all_chunks)}",
            system=build_system_prompt(all_chunks, detailed=True),
        )

    for _step in range(AGENTIC_MAX_STEPS):
        agent_prompt = "\n".join(transcript) + "\n\nWhat is your next action?"
        decision = llm(agent_prompt, system=AGENTIC_SYSTEM)

        if decision.upper().startswith("ANSWER:"):
            final = decision.split(":", 1)[1].strip()
            actions_log.append("ANSWER")
            final = _expand_if_detailed(final)
            return {"answer": final, "chunks_used": all_chunks, "notes": f"agent steps: {actions_log}"}

        if decision.upper().startswith("SEARCH:"):
            search_q = decision.split(":", 1)[1].strip()
            actions_log.append(f"SEARCH({search_q})")
            chunks = store.vector_search(search_q, k=k)
            all_chunks.extend(c for c in chunks if c.id not in {x.id for x in all_chunks})
            transcript.append(f"Action: SEARCH: {search_q}")
            transcript.append(f"Results:\n{format_context(chunks)}")
        else:
            actions_log.append("UNPARSED->ANSWER")
            decision = _expand_if_detailed(decision)
            return {"answer": decision, "chunks_used": all_chunks, "notes": f"agent steps: {actions_log}"}

    wrap_system = "Answer using everything gathered above." if not detailed else build_system_prompt(all_chunks, detailed=True)
    wrap_up = llm(
        "\n".join(transcript) + f"\n\nYou've reached the search limit. "
        f"Give your best final answer to: {query}",
        system=wrap_system,
    )
    actions_log.append("FORCED_ANSWER")
    return {"answer": wrap_up, "chunks_used": all_chunks, "notes": f"agent steps: {actions_log}"}


# =================================================================
# STRATEGY 7 — MULTIMODAL RAG
# =================================================================

def mode_multimodal_rag(
    store: Store, query: str, memory: Memory, k: int = None,
    query_image_path: Optional[str] = None, detailed: bool = False, **kwargs,
) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    image_description = None
    if query_image_path:
        img_path = Path(query_image_path)
        if not img_path.exists():
            return {"answer": f"Image not found: {query_image_path}", "chunks_used": []}
        image_description = describe_image(img_path)

    search_text = f"{query}\n{image_description}" if image_description else query
    chunks = store.vector_search(search_text, k=k)
    context = format_context(chunks)

    if image_description:
        prompt = (
            f"The user attached an image. Vision-model description of that image:\n"
            f"{image_description}\n\n"
            f"Retrieved knowledge-base context:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        prompt = f"Context (may include descriptions of ingested images):\n{context}\n\nQuestion: {query}"

    answer = llm(prompt, system=build_system_prompt(chunks, detailed))
    notes = "query included an attached image" if image_description else "text-only query (ingested images are searchable too)"
    return {"answer": answer, "chunks_used": chunks, "notes": notes}


# =================================================================
# STRATEGY 8 — ADAPTIVE RAG
# =================================================================

ADAPTIVE_ROUTES = {
    "SIMPLE_FACTUAL": "simple",
    "MULTI_PART": "branched",
    "CONVERSATIONAL_FOLLOWUP": "memory",
    "AMBIGUOUS_OR_RISKY": "corrective",
    "RELATIONSHIP_OR_ENTITY": "graph",
}


def _classify_query(query: str, has_history: bool) -> str:
    options = ", ".join(ADAPTIVE_ROUTES.keys())
    prompt = (
        f"Classify this question into exactly one category from: {options}\n\n"
        f"- SIMPLE_FACTUAL: a single, clear, self-contained factual lookup\n"
        f"- MULTI_PART: comparisons, multi-part, or 'causes and effects' style questions\n"
        f"- CONVERSATIONAL_FOLLOWUP: relies on previous turns (pronouns like 'it', 'that', 'them')\n"
        f"- AMBIGUOUS_OR_RISKY: vague, broad, or where wrong/unsupported answers would be costly\n"
        f"- RELATIONSHIP_OR_ENTITY: asks how people/things/organisations relate to each other\n\n"
        f"Question: \"{query}\"\n"
        f"{'(Note: there IS prior conversation history.)' if has_history else '(Note: there is NO prior conversation history.)'}\n\n"
        "Reply with exactly one category name, nothing else."
    )
    raw = llm(prompt).strip().upper()
    for key in ADAPTIVE_ROUTES:
        if key in raw:
            return key
    return "SIMPLE_FACTUAL"


def mode_adaptive_rag(store: Store, query: str, memory: Memory, k: int = None, detailed: bool = False, **kwargs) -> Dict[str, Any]:
    k = k if k is not None else config.TOP_K
    category = _classify_query(query, has_history=bool(memory.turns))
    route = ADAPTIVE_ROUTES[category]

    if route == "simple":
        result = mode_simple_rag(store, query, memory, k=k, detailed=detailed)
    elif route == "branched":
        result = mode_branched_rag(store, query, memory, k=k, detailed=detailed)
    elif route == "memory":
        result = mode_simple_rag_memory(store, query, memory, k=k, detailed=detailed)
    elif route == "corrective":
        result = mode_corrective_rag(store, query, memory, k=k, detailed=detailed)
    elif route == "graph":
        result = mode_graph_rag(store, query, memory, k=k, detailed=detailed)
    else:
        result = mode_simple_rag(store, query, memory, k=k, detailed=detailed)

    result["notes"] = f"classified as {category} -> routed to '{route}' mode. " + result.get("notes", "")
    return result


# =================================================================
# MODE REGISTRY — every strategy, addressable by number or name
# =================================================================

MODES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "1": mode_simple_rag, "simple": mode_simple_rag,
    "2": mode_simple_rag_memory, "memory": mode_simple_rag_memory,
    "3": mode_agentic_rag, "agentic": mode_agentic_rag,
    "4": mode_graph_rag, "graph": mode_graph_rag,
    "5": mode_self_rag, "self": mode_self_rag, "self-rag": mode_self_rag,
    "6": mode_branched_rag, "branched": mode_branched_rag,
    "7": mode_multimodal_rag, "multimodal": mode_multimodal_rag,
    "8": mode_adaptive_rag, "adaptive": mode_adaptive_rag,
    "9": mode_speculative_rag, "speculative": mode_speculative_rag,
    "10": mode_corrective_rag, "corrective": mode_corrective_rag, "crag": mode_corrective_rag,
    "11": mode_modular_rag, "modular": mode_modular_rag,
    "12": mode_naive_rag, "naive": mode_naive_rag,
    "13": mode_advanced_rag, "advanced": mode_advanced_rag,
    "14": mode_hyde, "hyde": mode_hyde,
}

MODE_LABELS: List[Tuple[str, str]] = [
    ("1", "Simple RAG (original)"),
    ("2", "Simple RAG with memory"),
    ("3", "Agentic RAG"),
    ("4", "Graph RAG"),
    ("5", "Self-RAG"),
    ("6", "Branched RAG"),
    ("7", "Multimodal RAG"),
    ("8", "Adaptive RAG"),
    ("9", "Speculative RAG"),
    ("10", "Corrective RAG"),
    ("11", "Modular RAG"),
    ("12", "Naive RAG"),
    ("13", "Advanced RAG"),
    ("14", "HyDE (hypothetical document embedding)"),
]


def resolve_mode(name: str) -> Tuple[str, Callable]:
    key = name.strip().lower()
    if key not in MODES:
        raise ValueError(
            f"Unknown mode '{name}'. Use a number 1-14 or a name "
            f"(simple, memory, agentic, graph, self, branched, multimodal, "
            f"adaptive, speculative, corrective, modular, naive, advanced, hyde)."
        )
    return key, MODES[key]


def run_mode(mode_name: str, store: Store, query: str, memory: Memory, **kwargs) -> Dict[str, Any]:
    _, fn = resolve_mode(mode_name)
    return fn(store, query, memory, **kwargs)
