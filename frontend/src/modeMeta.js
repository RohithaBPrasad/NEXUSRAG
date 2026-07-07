// Ground-truth pipeline stages per mode, mirroring what backend/app/modes.py
// actually does for each strategy. Used to render the pipeline strip so a
// user can see, at a glance, how a given RAG architecture actually works
// before they run it.
//
// stage type -> css class: 'retrieve' | 'transform' | 'generate'

export const MODE_PIPELINES = {
  '1': [
    { label: 'Retrieve (top-k)', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
  '2': [
    { label: 'History', type: 'transform' },
    { label: 'Retrieve (top-k)', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
  '3': [
    { label: 'Decide: search / answer', type: 'transform' },
    { label: 'Search (loop, max 4)', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
  '4': [
    { label: 'Match entities', type: 'transform' },
    { label: 'Traverse graph', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
  '5': [
    { label: 'Needs retrieval?', type: 'transform' },
    { label: 'Retrieve', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
    { label: 'Self-critique', type: 'transform' },
    { label: 'Retry (if ungrounded)', type: 'generate' },
  ],
  '6': [
    { label: 'Decompose query', type: 'transform' },
    { label: 'Retrieve × N branches', type: 'retrieve' },
    { label: 'Synthesize', type: 'generate' },
  ],
  '7': [
    { label: 'Describe image (vision)', type: 'transform' },
    { label: 'Retrieve', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
  '8': [
    { label: 'Classify query', type: 'transform' },
    { label: 'Route to sub-mode', type: 'transform' },
    { label: 'Run routed pipeline', type: 'generate' },
  ],
  '9': [
    { label: 'Draft (k=1)', type: 'retrieve' },
    { label: 'Draft answer', type: 'generate' },
    { label: 'Verify (k×2)', type: 'retrieve' },
    { label: 'Final answer', type: 'generate' },
  ],
  '10': [
    { label: 'Retrieve', type: 'retrieve' },
    { label: 'Grade relevance', type: 'transform' },
    { label: 'Rewrite + retry (if weak)', type: 'transform' },
    { label: 'Generate', type: 'generate' },
  ],
  '11': [
    { label: 'Retriever module', type: 'retrieve' },
    { label: 'Reranker module', type: 'transform' },
    { label: 'Generate', type: 'generate' },
  ],
  '12': [
    { label: 'Retrieve (top-1)', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
  '13': [
    { label: 'Rewrite query', type: 'transform' },
    { label: 'Hybrid retrieve', type: 'retrieve' },
    { label: 'LLM rerank', type: 'transform' },
    { label: 'Generate', type: 'generate' },
  ],
  '14': [
    { label: 'Hypothesize answer', type: 'generate' },
    { label: 'Embed hypothesis', type: 'transform' },
    { label: 'Retrieve by similarity', type: 'retrieve' },
    { label: 'Generate', type: 'generate' },
  ],
}

export const MODE_DESCRIPTIONS = {
  '1': 'Textbook baseline: embed the query, pull top-k chunks, stuff into the prompt.',
  '2': 'Simple RAG plus a rolling conversation window so follow-ups resolve correctly.',
  '3': 'The model chooses SEARCH or ANSWER in a loop, issuing its own refined queries.',
  '4': 'Retrieves by traversing a lightweight entity graph instead of pure similarity.',
  '5': 'Decides for itself whether retrieval is needed, then critiques its own groundedness.',
  '6': 'Splits multi-part questions into sub-questions, retrieves per-branch, then synthesizes.',
  '7': 'Ingested images are described by a vision model; a query image is handled the same way.',
  '8': 'Classifies the query, then routes it to whichever other mode fits best.',
  '9': 'A fast narrow draft is generated first, then verified/corrected against a wider retrieval.',
  '10': 'Grades each chunk for relevance; rewrites and retries retrieval if too many fail.',
  '11': 'Explicit swappable pipeline: pick a retriever module and a reranker module.',
  '12': 'Bare minimum baseline: fixed top-1 chunk, no reranking, no citation.',
  '13': 'Query rewriting, hybrid (vector + keyword) search, then LLM reranking.',
  '14': 'Writes a hypothetical answer first, embeds THAT, and retrieves using its embedding.',
}
