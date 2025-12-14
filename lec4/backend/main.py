import sys
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    backend_dir = Path(__file__).parent
    parent_dir = backend_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import google.genai as genai
import os
from dotenv import load_dotenv

from backend.database import get_db, init_db
from backend.models import DocumentModel

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await init_db()
    yield
    # Shutdown (if needed in the future)


app = FastAPI(
    title="Vector Database Demo",
    description="Demonstration of vector database retrieval techniques",
    lifespan=lifespan,
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Generative AI client
api_key = os.getenv("GEMINI_API_KEY")
genai_client = None
if api_key:
    genai_client = genai.Client(api_key=api_key)


# Pydantic models
class DocumentCreate(BaseModel):
    """Request model for creating a document"""

    content: str


class Document(BaseModel):
    """Response model for a document"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str


class DocumentWithSimilarity(Document):
    """Document with similarity score"""

    similarity: float


class SearchRequest(BaseModel):
    """Request model for searching documents"""

    query: str
    limit: int = 5


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using Google Generative AI

    Args:
        text: The text to generate an embedding for
        task_type: The task type - "retrieval_document" for documents,
                   "retrieval_query" for search queries

    Returns:
        List of floats representing the embedding vector
    """
    if not genai_client:
        raise Exception("Google API key is not set")

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
        # The embedding object should have a 'values' attribute containing a list of floats
        if not hasattr(embedding_obj, "values"):
            raise Exception(
                f"Embedding object missing 'values' attribute. Type: {type(embedding_obj)}, Attributes: {[x for x in dir(embedding_obj) if not x.startswith('_')]}"
            )

        embedding_values = embedding_obj.values

        # Validate the embedding
        if embedding_values is None:
            raise Exception("Embedding values is None")

        # Convert to list - handle different types (list, tuple, numpy array, etc.)
        try:
            import numpy as np

            if isinstance(embedding_values, np.ndarray):
                embedding_list = embedding_values.tolist()
            else:
                embedding_list = list(embedding_values)
        except ImportError:
            # If numpy is not available, try direct conversion
            if isinstance(embedding_values, (list, tuple)):
                embedding_list = list(embedding_values)
            else:
                # Try to iterate and convert
                embedding_list = [float(v) for v in embedding_values]

        if len(embedding_list) == 0:
            raise Exception("Embedding values is empty after conversion")

        # Convert all values to floats to ensure proper type
        embedding_list = [float(v) for v in embedding_list]

        # Check if all zeros (which would indicate a problem)
        if all(
            abs(v) < 1e-10 for v in embedding_list
        ):  # Use small epsilon for float comparison
            raise Exception(
                f"Warning: All embedding values are effectively zero (first 5: {embedding_list[:5]})! This indicates a problem with the API response."
            )

        return embedding_list

    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")


@app.post("/api/documents", response_model=Document)
async def create_document(doc: DocumentCreate, db: AsyncSession = Depends(get_db)):
    """POST endpoint to create a new document with embedding"""
    # Generate embedding for the document content
    embedding = get_embedding(doc.content)

    # Debug: verify embedding before storing
    print(f"DEBUG: Generated embedding length: {len(embedding)}")
    print(f"DEBUG: First 5 values: {embedding[:5]}")
    print(f"DEBUG: All zeros check: {all(v == 0.0 for v in embedding)}")

    new_doc = DocumentModel(content=doc.content, embedding=embedding)
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # Debug: verify embedding after storing
    if hasattr(new_doc, "embedding") and new_doc.embedding is not None:
        # Convert embedding to list for inspection
        if hasattr(new_doc.embedding, "__iter__"):
            stored_embedding = list(new_doc.embedding)
            print(f"DEBUG: Stored embedding length: {len(stored_embedding)}")
            print(f"DEBUG: Stored first 5 values: {stored_embedding[:5]}")
            print(
                f"DEBUG: Stored all zeros check: {all(v == 0.0 for v in stored_embedding)}"
            )

    return Document.model_validate(new_doc)


@app.get("/api/documents", response_model=List[Document])
async def get_documents(db: AsyncSession = Depends(get_db)):
    """GET endpoint to retrieve all documents"""
    result = await db.execute(select(DocumentModel))
    documents = result.scalars().all()
    return [Document.model_validate(doc) for doc in documents]


@app.post("/api/documents/search", response_model=List[DocumentWithSimilarity])
async def search_documents(search: SearchRequest, db: AsyncSession = Depends(get_db)):
    """Search for similar documents using vector similarity"""
    # Generate embedding for the search query
    query_embedding = get_embedding(search.query)
    print(f"DEBUG: Query embedding length: {len(query_embedding)}")
    print(f"DEBUG: Query embedding first 5 values: {query_embedding[:5]}")

    # Use SQLAlchemy ORM with pgvector's cosine_distance method
    # cosine_distance returns a value between 0 and 2 (0 = identical, 2 = opposite)
    # similarity = 1 - cosine_distance (normalized to 0-1 range)
    cosine_distance = DocumentModel.embedding.cosine_distance(query_embedding)
    similarity = 1 - cosine_distance

    # Build query using SQLAlchemy ORM
    stmt = (
        select(
            DocumentModel.id,
            DocumentModel.content,
            similarity.label("similarity"),
        )
        .where(DocumentModel.embedding.isnot(None))
        .order_by(cosine_distance)
    )

    try:
        result = await db.execute(stmt)
        rows = result.all()
        print(f"DEBUG: Found {len(rows)} results")

        documents = [
            DocumentWithSimilarity(
                id=row.id,
                content=row.content,
                similarity=float(row.similarity) if row.similarity is not None else 0.0,
            )
            for row in rows
        ]

        return documents
    except Exception as e:
        print(f"ERROR in search query: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)
