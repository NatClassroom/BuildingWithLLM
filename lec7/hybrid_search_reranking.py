"""
Hybrid Search with Reranking Demo

This script demonstrates hybrid search by combining:
1. Sparse embeddings (TF-IDF) - good for keyword matching
2. Dense embeddings (Gemini) - good for semantic understanding
3. Reranking - using a more sophisticated model to refine results

The hybrid approach combines the strengths of both sparse and dense methods.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import os
import warnings

warnings.filterwarnings("ignore")

# Initialize Google Generative AI client for embeddings
try:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai_client = genai.Client(api_key=api_key)
        HAS_GEMINI = True
    else:
        genai_client = None
        HAS_GEMINI = False
        print("Warning: GEMINI_API_KEY not set. Dense embeddings will not work.")
        print("Set it with: export GEMINI_API_KEY=your_api_key")
except ImportError:
    genai_client = None
    HAS_GEMINI = False
    print("Warning: google-genai not available. Install with: pip install google-genai")


def create_sample_documents() -> List[str]:
    """Create sample documents for demonstration."""
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Natural language processing helps computers understand and generate human language.",
        "Vector embeddings represent text as dense numerical vectors in high-dimensional space.",
        "Retrieval systems use embeddings to find relevant documents based on semantic similarity.",
        "TF-IDF is a traditional sparse embedding method that weights terms by their frequency and rarity.",
        "Hybrid search combines multiple retrieval methods to improve search accuracy.",
        "Reranking improves search results by using more sophisticated models to score candidates.",
        "Cross-encoders are powerful reranking models that process query-document pairs together.",
        "BM25 is another sparse retrieval method that improves upon TF-IDF for search tasks.",
    ]
    return documents


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using Google Generative AI (Gemini).

    Args:
        text: The text to generate an embedding for

    Returns:
        List of floats representing the embedding vector
    """
    if not genai_client:
        raise Exception(
            "Gemini API client not initialized. Set GEMINI_API_KEY environment variable."
        )

    try:
        # Use the client to generate embeddings
        result = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )

        # Check if result has embeddings
        if not hasattr(result, "embeddings"):
            raise Exception(
                f"API response missing 'embeddings' attribute. Available attributes: {[x for x in dir(result) if not x.startswith('_')]}"
            )

        if not result.embeddings or len(result.embeddings) == 0:
            raise Exception("No embeddings returned from API")

        # Get the first embedding object
        embedding_obj = result.embeddings[0]

        # Extract the embedding values
        if not hasattr(embedding_obj, "values"):
            raise Exception(
                f"Embedding object missing 'values' attribute. Type: {type(embedding_obj)}, Attributes: {[x for x in dir(embedding_obj) if not x.startswith('_')]}"
            )

        embedding_values = embedding_obj.values

        # Validate the embedding
        if embedding_values is None:
            raise Exception("Embedding values is None")

        # Convert to list
        if isinstance(embedding_values, np.ndarray):
            embedding_list = embedding_values.tolist()
        elif isinstance(embedding_values, (list, tuple)):
            embedding_list = list(embedding_values)
        else:
            embedding_list = [float(v) for v in embedding_values]

        if len(embedding_list) == 0:
            raise Exception("Embedding values is empty after conversion")

        # Convert all values to floats
        embedding_list = [float(v) for v in embedding_list]

        return embedding_list

    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")


def get_dense_embeddings(texts: List[str], model=None):
    """
    Get dense embeddings for texts using Gemini.

    Args:
        texts: List of text strings
        model: Not used (kept for compatibility)

    Returns:
        numpy array of embeddings
    """
    if not HAS_GEMINI:
        raise Exception(
            "Gemini API not available. Set GEMINI_API_KEY environment variable."
        )

    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)

    return np.array(embeddings)


def sparse_retrieval(
    documents: List[str],
    query: str,
    vectorizer: TfidfVectorizer = None,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Perform sparse (TF-IDF) retrieval.

    Returns:
        List of (document_index, score) tuples
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        doc_vectors = vectorizer.fit_transform(documents)
    else:
        doc_vectors = vectorizer.transform(documents)

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(idx, float(similarities[idx])) for idx in top_indices]


def dense_retrieval(
    documents: List[str],
    query: str,
    doc_embeddings: np.ndarray = None,
    query_embedding: np.ndarray = None,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Perform dense embedding retrieval using Gemini.

    Returns:
        List of (document_index, score) tuples
    """
    if doc_embeddings is None:
        doc_embeddings = get_dense_embeddings(documents)

    if query_embedding is None:
        query_emb = get_dense_embeddings([query])
        query_embedding = query_emb[0]

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(idx, float(similarities[idx])) for idx in top_indices]


def hybrid_search(
    documents: List[str],
    query: str,
    sparse_weight: float = 0.3,
    dense_weight: float = 0.7,
    top_k: int = 10,
    vectorizer: TfidfVectorizer = None,
    doc_embeddings: np.ndarray = None,
) -> List[Tuple[int, float]]:
    """
    Perform hybrid search by combining sparse and dense retrieval.

    Args:
        documents: List of document texts
        query: Search query
        sparse_weight: Weight for sparse (TF-IDF) scores
        dense_weight: Weight for dense embedding scores
        top_k: Number of results to return

    Returns:
        List of (document_index, combined_score) tuples
    """
    # Get sparse results
    sparse_results = sparse_retrieval(
        documents, query, vectorizer, top_k=len(documents)
    )
    sparse_scores = {idx: score for idx, score in sparse_results}

    # Get dense results
    dense_results = dense_retrieval(
        documents, query, doc_embeddings, top_k=len(documents)
    )
    dense_scores = {idx: score for idx, score in dense_results}

    # Normalize scores to [0, 1] range for fair combination
    if sparse_results:
        max_sparse = max(score for _, score in sparse_results)
        min_sparse = min(score for _, score in sparse_results)
        sparse_range = max_sparse - min_sparse if max_sparse != min_sparse else 1
    else:
        sparse_range = 1

    if dense_results:
        max_dense = max(score for _, score in dense_results)
        min_dense = min(score for _, score in dense_results)
        dense_range = max_dense - min_dense if max_dense != min_dense else 1
    else:
        dense_range = 1

    # Combine scores
    combined_scores = {}
    all_indices = set(sparse_scores.keys()) | set(dense_scores.keys())

    for idx in all_indices:
        sparse_norm = (
            (sparse_scores.get(idx, 0) - min_sparse) / sparse_range
            if sparse_range > 0
            else 0
        )
        dense_norm = (
            (dense_scores.get(idx, 0) - min_dense) / dense_range
            if dense_range > 0
            else 0
        )

        combined = (sparse_weight * sparse_norm) + (dense_weight * dense_norm)
        combined_scores[idx] = combined

    # Sort by combined score
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in sorted_results[:top_k]]


def rerank_with_dense(
    documents: List[str], query: str, candidate_indices: List[int], model=None
) -> List[Tuple[int, float]]:
    """
    Rerank candidates using dense embeddings with Gemini.

    In practice, you would use a cross-encoder model that processes
    query-document pairs together. Here we use Gemini embeddings
    for reranking, which provides better semantic understanding.

    Args:
        documents: All documents
        query: Search query
        candidate_indices: Indices of candidate documents to rerank
        model: Not used (kept for compatibility)

    Returns:
        Reranked list of (document_index, score) tuples
    """
    if not HAS_GEMINI:
        raise Exception(
            "Gemini API not available. Set GEMINI_API_KEY environment variable."
        )

    # Get embeddings for query and candidates
    candidate_docs = [documents[idx] for idx in candidate_indices]

    # Get embeddings using Gemini
    query_emb = get_dense_embeddings([query])[0]
    doc_embs = get_dense_embeddings(candidate_docs)

    # Calculate similarities
    similarities = cosine_similarity([query_emb], doc_embs)[0]

    # Create results with original indices
    results = [
        (candidate_indices[i], float(similarities[i]))
        for i in range(len(candidate_indices))
    ]

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def demonstrate_hybrid_search():
    """Main demonstration function."""
    print("=" * 80)
    print("Hybrid Search with Reranking Demo")
    print("=" * 80)
    print()

    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    print()

    # Pre-compute embeddings for efficiency
    print("Pre-computing embeddings...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)

    if HAS_GEMINI:
        print("Generating Gemini embeddings for documents...")
        doc_embeddings = get_dense_embeddings(documents)
        print(f"Generated embeddings with dimension: {doc_embeddings.shape[1]}")
    else:
        print("Warning: Gemini API not available. Dense retrieval will not work.")
        doc_embeddings = None

    print("Done!")
    print()

    # Test query
    query = "How do we find relevant documents using embeddings?"
    print("=" * 80)
    print(f"Query: '{query}'")
    print("=" * 80)
    print()

    # 1. Sparse retrieval only
    print("1. SPARSE RETRIEVAL (TF-IDF) ONLY")
    print("-" * 80)
    sparse_results = sparse_retrieval(documents, query, vectorizer, top_k=5)
    for rank, (idx, score) in enumerate(sparse_results, 1):
        print(f"Rank {rank} (Score: {score:.4f}): Doc {idx + 1}")
        print(f"  {documents[idx]}")
    print()

    # 2. Dense retrieval only
    print("2. DENSE RETRIEVAL (Gemini Embeddings) ONLY")
    print("-" * 80)
    if doc_embeddings is not None:
        dense_results = dense_retrieval(documents, query, doc_embeddings, top_k=5)
        for rank, (idx, score) in enumerate(dense_results, 1):
            print(f"Rank {rank} (Score: {score:.4f}): Doc {idx + 1}")
            print(f"  {documents[idx]}")
    else:
        print("Skipped: Gemini API not available")
    print()

    # 3. Hybrid search
    print("3. HYBRID SEARCH (Combining Sparse + Dense)")
    print("-" * 80)
    print("Weights: Sparse=0.3, Dense=0.7")
    hybrid_results = []
    if doc_embeddings is not None:
        hybrid_results = hybrid_search(
            documents,
            query,
            sparse_weight=0.3,
            dense_weight=0.7,
            top_k=5,
            vectorizer=vectorizer,
            doc_embeddings=doc_embeddings,
        )
        for rank, (idx, score) in enumerate(hybrid_results, 1):
            print(f"Rank {rank} (Combined Score: {score:.4f}): Doc {idx + 1}")
            print(f"  {documents[idx]}")
    else:
        print("Skipped: Gemini API not available")
    print()

    # 4. Hybrid search with reranking
    print("4. HYBRID SEARCH + RERANKING")
    print("-" * 80)
    if doc_embeddings is not None and len(hybrid_results) > 0:
        print("Step 1: Get top 10 candidates from hybrid search")
        candidate_indices = [idx for idx, _ in hybrid_results[:10]]
        print(f"Candidates: {[idx + 1 for idx in candidate_indices]}")
        print()

        print("Step 2: Rerank candidates using Gemini embeddings")
        reranked_results = rerank_with_dense(documents, query, candidate_indices)
        for rank, (idx, score) in enumerate(reranked_results[:5], 1):
            print(f"Rank {rank} (Reranked Score: {score:.4f}): Doc {idx + 1}")
            print(f"  {documents[idx]}")
    else:
        print("Skipped: Gemini API not available or no hybrid results")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        """
Hybrid search combines the strengths of:
- Sparse embeddings (TF-IDF): Excellent for exact keyword matching
- Dense embeddings: Better for semantic understanding and synonyms

Reranking improves results by:
- Using more sophisticated models (e.g., cross-encoders)
- Processing query-document pairs together
- Refining the top candidates from initial retrieval

This two-stage approach (retrieval + reranking) is common in production
search systems because it balances speed and accuracy.
    """
    )


if __name__ == "__main__":
    demonstrate_hybrid_search()
