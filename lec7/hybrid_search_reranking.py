"""
Hybrid Search with Reranking Demo

This script demonstrates hybrid search by combining:
1. Sparse embeddings (TF-IDF) - good for keyword matching
2. Dense embeddings (sentence transformers) - good for semantic understanding
3. Reranking - using a more sophisticated model to refine results

The hybrid approach combines the strengths of both sparse and dense methods.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence-transformers, fallback to simple dense embeddings if not available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available. Using simple dense embeddings.")
    print("Install with: pip install sentence-transformers")


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


def get_dense_embeddings(texts: List[str], model=None):
    """
    Get dense embeddings for texts.
    
    Args:
        texts: List of text strings
        model: Optional pre-loaded sentence transformer model
    
    Returns:
        numpy array of embeddings
    """
    if HAS_SENTENCE_TRANSFORMERS:
        if model is None:
            # Use a lightweight model for demonstration
            model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts, show_progress_bar=False)
    else:
        # Fallback: Simple dense embeddings using TF-IDF with more features
        # This is not ideal but demonstrates the concept
        vectorizer = TfidfVectorizer(
            max_features=384,  # Match typical embedding dimension
            stop_words='english'
        )
        return vectorizer.fit_transform(texts).toarray()


def sparse_retrieval(
    documents: List[str],
    query: str,
    vectorizer: TfidfVectorizer = None,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Perform sparse (TF-IDF) retrieval.
    
    Returns:
        List of (document_index, score) tuples
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
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
    model=None,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Perform dense embedding retrieval.
    
    Returns:
        List of (document_index, score) tuples
    """
    if doc_embeddings is None:
        doc_embeddings = get_dense_embeddings(documents, model)
    
    if query_embedding is None:
        query_emb = get_dense_embeddings([query], model)
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
    model=None
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
    sparse_results = sparse_retrieval(documents, query, vectorizer, top_k=len(documents))
    sparse_scores = {idx: score for idx, score in sparse_results}
    
    # Get dense results
    dense_results = dense_retrieval(documents, query, doc_embeddings, model=model, top_k=len(documents))
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
        sparse_norm = (sparse_scores.get(idx, 0) - min_sparse) / sparse_range if sparse_range > 0 else 0
        dense_norm = (dense_scores.get(idx, 0) - min_dense) / dense_range if dense_range > 0 else 0
        
        combined = (sparse_weight * sparse_norm) + (dense_weight * dense_norm)
        combined_scores[idx] = combined
    
    # Sort by combined score
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in sorted_results[:top_k]]


def rerank_with_dense(
    documents: List[str],
    query: str,
    candidate_indices: List[int],
    model=None
) -> List[Tuple[int, float]]:
    """
    Rerank candidates using dense embeddings (simulating a cross-encoder).
    
    In practice, you would use a cross-encoder model that processes
    query-document pairs together. Here we use a more sophisticated
    dense embedding model for demonstration.
    
    Args:
        documents: All documents
        query: Search query
        candidate_indices: Indices of candidate documents to rerank
        model: Optional pre-loaded model
    
    Returns:
        Reranked list of (document_index, score) tuples
    """
    if model is None and HAS_SENTENCE_TRANSFORMERS:
        # Use a better model for reranking
        model = SentenceTransformer('all-mpnet-base-v2')
    
    # Get embeddings for query and candidates
    candidate_docs = [documents[idx] for idx in candidate_indices]
    
    # Create query-document pairs for better semantic matching
    query_emb = get_dense_embeddings([query], model)[0]
    doc_embs = get_dense_embeddings(candidate_docs, model)
    
    # Calculate similarities
    similarities = cosine_similarity([query_emb], doc_embs)[0]
    
    # Create results with original indices
    results = [(candidate_indices[i], float(similarities[i])) 
               for i in range(len(candidate_indices))]
    
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
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    
    model = None
    if HAS_SENTENCE_TRANSFORMERS:
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    doc_embeddings = get_dense_embeddings(documents, model)
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
    print("2. DENSE RETRIEVAL (Semantic Embeddings) ONLY")
    print("-" * 80)
    dense_results = dense_retrieval(documents, query, doc_embeddings, model=model, top_k=5)
    for rank, (idx, score) in enumerate(dense_results, 1):
        print(f"Rank {rank} (Score: {score:.4f}): Doc {idx + 1}")
        print(f"  {documents[idx]}")
    print()
    
    # 3. Hybrid search
    print("3. HYBRID SEARCH (Combining Sparse + Dense)")
    print("-" * 80)
    print("Weights: Sparse=0.3, Dense=0.7")
    hybrid_results = hybrid_search(
        documents, 
        query, 
        sparse_weight=0.3, 
        dense_weight=0.7,
        top_k=5,
        vectorizer=vectorizer,
        doc_embeddings=doc_embeddings,
        model=model
    )
    for rank, (idx, score) in enumerate(hybrid_results, 1):
        print(f"Rank {rank} (Combined Score: {score:.4f}): Doc {idx + 1}")
        print(f"  {documents[idx]}")
    print()
    
    # 4. Hybrid search with reranking
    print("4. HYBRID SEARCH + RERANKING")
    print("-" * 80)
    print("Step 1: Get top 10 candidates from hybrid search")
    candidate_indices = [idx for idx, _ in hybrid_results[:10]]
    print(f"Candidates: {[idx + 1 for idx in candidate_indices]}")
    print()
    
    print("Step 2: Rerank candidates using more sophisticated dense model")
    reranked_results = rerank_with_dense(documents, query, candidate_indices, model=model)
    for rank, (idx, score) in enumerate(reranked_results[:5], 1):
        print(f"Rank {rank} (Reranked Score: {score:.4f}): Doc {idx + 1}")
        print(f"  {documents[idx]}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Hybrid search combines the strengths of:
- Sparse embeddings (TF-IDF): Excellent for exact keyword matching
- Dense embeddings: Better for semantic understanding and synonyms

Reranking improves results by:
- Using more sophisticated models (e.g., cross-encoders)
- Processing query-document pairs together
- Refining the top candidates from initial retrieval

This two-stage approach (retrieval + reranking) is common in production
search systems because it balances speed and accuracy.
    """)


if __name__ == "__main__":
    demonstrate_hybrid_search()

