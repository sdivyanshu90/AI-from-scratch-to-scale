# Advanced Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) augments a language model with a
retrieval system that fetches relevant documents at inference time. Instead
of encoding all world knowledge in model weights (expensive, static), RAG
decomposes the problem: a retriever finds relevant text, and the LM reasons
over it. This enables knowledge updating without retraining, reduces
hallucination on factual queries, and allows citation of sources.

## 1. The Core Intuition (The "Why")

Large language models hallucinate because they interpolate over training
data patterns rather than looking up facts from a verified source. Lewis
et al. (2020) showed that augmenting a model with a dense retriever allowed
it to answer open-domain questions with higher accuracy than a model twice
its size, while also being updatable (new documents are added to the index
without retraining).

The basic RAG pipeline is simple: embed the query, find the top-$k$ most
similar documents in a vector database, concatenate them with the query
in the LM's context window, and generate the answer.

Advanced RAG addresses the failure modes of this naive approach: what if the
relevant information spans multiple documents? What if the retrieved chunk
lacks the surrounding context needed to answer the question? What if the
query requires reasoning over a knowledge graph rather than text retrieval?

## 2. The Theoretical Underpinning

### Dense Retrieval

Dense retrieval (Karpukhin et al., 2020 — DPR) encodes queries and documents
into a shared dense vector space and computes similarity by dot product:

$$\text{sim}(q, d) = E_Q(q)^{\top} E_D(d)$$

where $E_Q$ and $E_D$ are encoder functions (typically BERT-style models).
The top-$k$ documents are retrieved via approximate nearest neighbour search
(FAISS, ScaNN, HNSW).

### Bi-Encoder vs Cross-Encoder

A **bi-encoder** encodes query and document independently:
$$s = E_Q(q)^{\top} E_D(d)$$
This is fast (O(1) per query given pre-encoded documents) and scalable to
millions of documents.

A **cross-encoder** concatenates query and document:
$$s = f(q \oplus d)$$
This is much more accurate (the model attends over both simultaneously) but
requires forward passes for each (query, document) pair, making it too slow
for first-stage retrieval over large corpora.

The standard architecture is a **two-stage pipeline**:
1. Bi-encoder retrieves top-100 candidates in milliseconds.
2. Cross-encoder re-ranks the top-100 to top-10 with much higher accuracy.

### Chunking Strategy

How you split documents into chunks critically affects retrieval quality:
- **Fixed-size chunks** (e.g., 512 tokens with 64-token overlap): simple but
  may split sentences or break context.
- **Sentence/paragraph chunking**: preserves semantic units.
- **Hierarchical chunking**: index small chunks for retrieval but retrieve
  the parent paragraph for context (Kamradt, 2023).
- **Semantic chunking**: split when the cosine similarity between adjacent
  sentence embeddings drops below a threshold.

### HyDE: Hypothetical Document Embeddings

Gao et al. (2022) showed that generating a hypothetical document that would
answer the query, then using its embedding for retrieval, outperforms using
the query embedding directly for knowledge-intensive tasks:

$$\text{embedding} = E_D(\text{LLM}(\text{"Answer: "} + q))$$

The reasoning: the query is typically a question; documents are statements.
The hypothetical document bridges the distributional gap.

### GraphRAG: Knowledge Graph-Augmented Retrieval

Edge et al. (2024) showed that for questions requiring synthesis across many
documents (global queries), a knowledge graph with community summaries
outperforms vector retrieval. GraphRAG extracts entities and relationships,
clusters them into communities, and generates community summaries, which are
retrieved via hierarchical traversal rather than dense vector search.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import re
from typing import Callable


def chunk_by_sentence(
    text:         str,
    max_tokens:   int = 512,
    overlap_sents: int = 1,
) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Args:
        text:         Input text.
        max_tokens:   Approximate max tokens per chunk (uses word count as proxy).
        overlap_sents: Number of sentences to overlap between chunks.

    Returns:
        List of text chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        words = len(sent.split())
        if current_len + words > max_tokens and current:
            chunks.append(" ".join(current))
            # Keep last overlap_sents sentences for context continuity
            current     = current[-overlap_sents:]
            current_len = sum(len(s.split()) for s in current)
        current.append(sent)
        current_len += words

    if current:
        chunks.append(" ".join(current))
    return chunks


class SimpleVectorStore:
    """In-memory vector store using cosine similarity (no external libs)."""

    def __init__(self) -> None:
        self.embeddings: list = []
        self.documents:  list[str] = []

    def add(self, doc: str, embedding: list[float]) -> None:
        import numpy as np
        v = np.array(embedding, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        self.embeddings.append(v)
        self.documents.append(doc)

    def search(self, query_embedding: list[float], k: int = 5) -> list[str]:
        import numpy as np
        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = [float(q @ e) for e in self.embeddings]
        top_k = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k]
```

## 4. Implementation: Production-Grade (LangChain / LlamaIndex)

```python
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_rag_pipeline(
    documents:      list[str],
    llm,
    embed_model:    str = "BAAI/bge-small-en-v1.5",
    chunk_size:     int = 512,
    chunk_overlap:  int = 64,
    top_k:          int = 5,
):
    """Build a standard RAG pipeline using FAISS and HuggingFace embeddings.

    Uses recursive character splitting, which respects sentence and
    paragraph boundaries before splitting on characters.

    Args:
        documents:  Raw text strings to index.
        llm:        LangChain-compatible LLM (e.g., ChatOpenAI).
        embed_model: HuggingFace embedding model name.
        chunk_size:  Target chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks.
        top_k:      Number of chunks to retrieve.

    Returns:
        Runnable RAG chain.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    docs       = splitter.create_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever  = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question.\n"
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def rerank_with_cross_encoder(
    query:     str,
    documents: list[str],
    top_k:     int = 5,
) -> list[str]:
    """Re-rank documents using a cross-encoder model.

    Uses sentence-transformers cross-encoder which jointly attends over
    the (query, document) pair for much higher accuracy than bi-encoder.

    Args:
        query:     The user's question.
        documents: Candidate documents from first-stage retrieval.
        top_k:     Number of documents to return after re-ranking.

    Returns:
        Top-k re-ranked documents.
    """
    from sentence_transformers import CrossEncoder
    model  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs  = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using overly large chunk sizes (> 1024 tokens). Large chunks
  may retrieve correctly but dilute the relevant passage within a sea of
  irrelevant context, confusing the LM.

  **Fix:** Use 256–512 token chunks. Use hierarchical chunking: retrieve
  small chunks for precision, expand to parent paragraphs for context.

- **Mistake:** Not persisting the vector index. Rebuilding FAISS on every
  application start for a large corpus takes minutes.

  **Fix:** Save the index with `vectorstore.save_local("index_dir")` and
  load it with `FAISS.load_local(...)` on startup.

- **Mistake:** Embedding queries with a model different from the one used
  to embed documents. Embeddings from different models are not comparable.

  **Fix:** Use the same embedding model for both indexing and querying. If
  you change the model, rebuild the entire index.

- **Mistake:** Not filtering retrieved documents by relevance score.
  Including irrelevant chunks in the context degrades generation quality.

  **Fix:** Set a minimum similarity threshold. LangChain supports
  `search_type="similarity_score_threshold"` in the retriever.

- **Mistake:** Using RAG for tasks that require precise recall of a specific
  fact across a million documents but only retrieving top-5 chunks.

  **Fix:** Tune retrieval recall on a representative question set. Use
  a cross-encoder re-ranker to improve precision at small $k$.

## 6. Knowledge Check

1. **Conceptual:** Explain the bi-encoder vs cross-encoder trade-off for
   document retrieval. Why can't you use a cross-encoder for first-stage
   retrieval over 10 million documents? At what stage is it appropriate?

2. **Conceptual:** HyDE uses a hypothetical document embedding instead of
   the query embedding for retrieval. What distributional gap does this
   address? Under what conditions might HyDE perform worse than direct
   query embedding?

3. **Coding challenge:** Build a RAG pipeline that answers questions about
   a set of research papers. Evaluate retrieval quality using recall@5:
   for each question, check if the gold-standard passage is in the top-5
   retrieved chunks. Compare fixed-size chunking vs sentence-based chunking.

## References

1. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*. arXiv:2005.11401.
2. Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*. arXiv:2004.04906.
3. Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise zero-shot dense retrieval without relevance labels. arXiv:2212.10496. *(HyDE.)*
4. Edge, D., Trinh, H., Cheng, N., et al. (2024). From local to global: A graph RAG approach to query-focused summarization. arXiv:2404.16130. *(GraphRAG.)*
