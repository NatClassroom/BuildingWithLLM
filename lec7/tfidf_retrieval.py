"""
TF-IDF Sparse Embedding Retrieval Demo

This script demonstrates how to use TF-IDF (Term Frequency-Inverse Document Frequency)
for sparse embedding retrieval. TF-IDF creates sparse vector representations where
most values are zero, capturing the importance of terms in documents.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


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
    ]
    return documents


def tfidf_retrieval(
    documents: List[str], 
    query: str, 
    top_k: int = 3
) -> List[Tuple[int, float, str]]:
    """
    Perform TF-IDF based retrieval.
    
    Args:
        documents: List of document texts
        query: Search query
        top_k: Number of top results to return
    
    Returns:
        List of tuples (document_index, similarity_score, document_text)
    """
    # Initialize TF-IDF vectorizer
    # max_features limits vocabulary size for demonstration
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    # Fit and transform documents
    print("Fitting TF-IDF vectorizer on documents...")
    doc_vectors = vectorizer.fit_transform(documents)
    
    # Transform query
    print(f"Transforming query: '{query}'")
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [
        (idx, float(similarities[idx]), documents[idx])
        for idx in top_indices
    ]
    
    return results


def demonstrate_tfidf():
    """Main demonstration function."""
    print("=" * 80)
    print("TF-IDF Sparse Embedding Retrieval Demo")
    print("=" * 80)
    print()
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    print()
    
    # Display documents
    print("Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")
    print()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are vector embeddings?",
        "Tell me about hybrid search",
    ]
    
    for query in test_queries:
        print("=" * 80)
        print(f"Query: '{query}'")
        print("=" * 80)
        
        results = tfidf_retrieval(documents, query, top_k=3)
        
        print(f"\nTop {len(results)} results:")
        for rank, (doc_idx, score, doc_text) in enumerate(results, 1):
            print(f"\nRank {rank} (Score: {score:.4f}):")
            print(f"  Document {doc_idx + 1}: {doc_text}")
        
        print()
    
    # Demonstrate sparsity
    print("=" * 80)
    print("Sparsity Demonstration")
    print("=" * 80)
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    
    # Calculate sparsity
    total_elements = doc_vectors.shape[0] * doc_vectors.shape[1]
    non_zero_elements = doc_vectors.nnz
    sparsity = 1 - (non_zero_elements / total_elements)
    
    print(f"Document vectors shape: {doc_vectors.shape}")
    print(f"Total elements: {total_elements:,}")
    print(f"Non-zero elements: {non_zero_elements:,}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"\nThis means {sparsity:.2%} of the vector values are zero!")
    print("This is why TF-IDF is called a 'sparse' embedding method.")


if __name__ == "__main__":
    demonstrate_tfidf()

