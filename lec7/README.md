# Hybrid Search and Reranking

This lecture demonstrates sparse embedding retrieval with TF-IDF and hybrid search with reranking.

## Scripts

### 1. `tfidf_retrieval.py` - Sparse Embedding Retrieval

Demonstrates TF-IDF (Term Frequency-Inverse Document Frequency) for sparse embedding retrieval.

**Features:**

- Creates sparse vector representations where most values are zero
- Uses TF-IDF to weight terms by their frequency and rarity
- Performs cosine similarity search
- Demonstrates the sparsity of TF-IDF vectors

**Usage:**

```bash
python lec7/tfidf_retrieval.py
```

**What it does:**

- Creates sample documents about machine learning and retrieval
- Shows how TF-IDF vectorizes documents and queries
- Performs retrieval for multiple test queries
- Demonstrates the sparsity of TF-IDF embeddings

### 2. `hybrid_search_reranking.py` - Hybrid Search with Reranking

Demonstrates hybrid search combining sparse (TF-IDF) and dense embeddings (Gemini), followed by reranking.

**Features:**

- Combines sparse (TF-IDF) and dense (Gemini) embeddings
- Shows how different retrieval methods complement each other
- Demonstrates reranking with Gemini embeddings
- Two-stage approach: retrieval + reranking

**Usage:**

```bash
python lec7/hybrid_search_reranking.py
```

**What it does:**

1. **Sparse Retrieval**: Shows TF-IDF results (good for keyword matching)
2. **Dense Retrieval**: Shows Gemini embedding results (good for semantic understanding)
3. **Hybrid Search**: Combines both methods with weighted scores
4. **Reranking**: Uses Gemini embeddings to refine top candidates

## Installation

The scripts require the following dependencies:

```bash
# Core dependencies (likely already installed)
pip install numpy scikit-learn

# For Gemini embeddings (required for hybrid_search_reranking.py)
# The google-genai package should already be in pyproject.toml
```

**Environment Setup:**

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY=your_google_api_key_here
```

Or create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_api_key_here
```

The hybrid search script requires the `GEMINI_API_KEY` to be set, as it uses Gemini's `text-embedding-004` model for dense embeddings.

## Key Concepts

### Sparse Embeddings (TF-IDF)

- **Sparse**: Most vector values are zero
- **Keyword-focused**: Matches exact terms
- **Fast**: Efficient for large vocabularies
- **Traditional**: Well-established method

### Dense Embeddings (Gemini)

- **Dense**: Most/all values are non-zero
- **Semantic**: Understands meaning and synonyms
- **Context-aware**: Captures relationships between words
- **Modern**: Uses neural network models (Gemini text-embedding-004)
- **API-based**: Uses Google's Gemini API for embedding generation

### Hybrid Search

- Combines sparse and dense methods
- Balances keyword matching and semantic understanding
- Uses weighted combination of scores
- Often outperforms either method alone

### Reranking

- Two-stage retrieval process
- Stage 1: Fast retrieval (hybrid search) to get candidates
- Stage 2: Slower but more accurate reranking of top candidates
- Uses Gemini embeddings to rerank candidates
- In production, you might use cross-encoders for even better results
- Common in production search systems

## Example Output

### TF-IDF Retrieval

Shows how sparse embeddings work and demonstrates that most vector values are zero.

### Hybrid Search

Shows how combining methods improves results:

- Sparse retrieval finds documents with exact keywords
- Dense retrieval finds semantically similar documents
- Hybrid search combines both strengths
- Reranking refines the final results

## Notes

- The scripts use sample documents for demonstration
- In production, you would use larger document collections
- Reranking models (like cross-encoders) are more computationally expensive
- The two-stage approach (retrieval + reranking) balances speed and accuracy
