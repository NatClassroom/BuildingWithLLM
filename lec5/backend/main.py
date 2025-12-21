import sys
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    backend_dir = Path(__file__).parent
    parent_dir = backend_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import google.genai as genai
from google.genai import errors as genai_errors
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
    title="RAG Demo API",
    description="Demonstration of RAG (Retrieval-Augmented Generation) process",
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
class Chunk(BaseModel):
    """Model for a text chunk"""

    text: str
    index: int


class ChunkWithEmbedding(Chunk):
    """Chunk with embedding vector"""

    embedding: List[float]


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


class ChatMessage(BaseModel):
    """Model for a chat message"""

    role: str  # "user" or "assistant"
    content: str
    retrieved_docs: Optional[List[DocumentWithSimilarity]] = None
    full_prompt: Optional[str] = None  # The complete prompt sent to the LLM


class ConversationMessage(BaseModel):
    """Model for a message in the conversation structure"""

    role: str  # "system", "user", or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for chat - frontend controls the full conversation structure"""

    system_instruction: Optional[str] = None  # System instruction with context
    messages: List[ConversationMessage]  # Full conversation: user and model messages
    limit: int = 3  # For document retrieval (if needed for response metadata)


class SearchRequest(BaseModel):
    """Request model for document search"""

    message: str
    limit: int = 3


class FileUploadResponse(BaseModel):
    """Response model for file upload"""

    filename: str
    text: str
    chunks: List[Chunk]


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""

    chunks: List[ChunkWithEmbedding]


class ConversationRequest(BaseModel):
    """Request model for getting conversation structure"""

    system_instruction: Optional[str] = None  # System instruction that was sent
    messages: List[ConversationMessage]  # Messages that were sent to the LLM
    retrieval_query: Optional[str] = None  # The query used for retrieval
    retrieved_documents: Optional[List[DocumentWithSimilarity]] = (
        None  # Retrieved chunks
    )
    base_system_instruction: Optional[str] = (
        None  # Base system instruction before augmentation
    )


class ConversationResponse(BaseModel):
    """Response model for conversation structure"""

    messages: List[ConversationMessage]  # The complete conversation structure
    retrieval_query: Optional[str] = None  # The query used for retrieval
    retrieved_documents: Optional[List[DocumentWithSimilarity]] = (
        None  # Retrieved chunks
    )
    base_system_instruction: Optional[str] = (
        None  # Base system instruction before augmentation
    )
    augmentation: Optional[str] = None  # How the system instruction was augmented


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using Google Generative AI

    Args:
        text: The text to generate an embedding for

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
        if not hasattr(embedding_obj, "values"):
            raise Exception(
                f"Embedding object missing 'values' attribute. Type: {type(embedding_obj)}, Attributes: {[x for x in dir(embedding_obj) if not x.startswith('_')]}"
            )

        embedding_values = embedding_obj.values

        # Validate the embedding
        if embedding_values is None:
            raise Exception("Embedding values is None")

        # Convert to list
        try:
            import numpy as np

            if isinstance(embedding_values, np.ndarray):
                embedding_list = embedding_values.tolist()
            else:
                embedding_list = list(embedding_values)
        except ImportError:
            if isinstance(embedding_values, (list, tuple)):
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


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Split text into chunks with overlap

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of Chunk objects
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, index=index))
        start = end - overlap
        index += 1

    return chunks


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a text file and extract text"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await file.read()
    text = content.decode("utf-8")

    # Chunk the text
    chunks = chunk_text(text)

    return FileUploadResponse(filename=file.filename, text=text, chunks=chunks)


@app.post("/api/embed", response_model=EmbeddingResponse)
async def embed_chunks(request: List[Chunk]):
    """Generate embeddings for chunks"""
    chunks_with_embeddings = []

    for chunk in request:
        embedding = get_embedding(chunk.text)
        chunks_with_embeddings.append(
            ChunkWithEmbedding(text=chunk.text, index=chunk.index, embedding=embedding)
        )

    return EmbeddingResponse(chunks=chunks_with_embeddings)


@app.post("/api/documents", response_model=List[Document])
async def store_documents(
    request: List[ChunkWithEmbedding], db: AsyncSession = Depends(get_db)
):
    """Store chunks with embeddings in the vector database"""
    documents = []

    for chunk in request:
        new_doc = DocumentModel(content=chunk.text, embedding=chunk.embedding)
        db.add(new_doc)
        documents.append(new_doc)

    await db.commit()

    # Refresh all documents to get their IDs
    for doc in documents:
        await db.refresh(doc)

    return [Document.model_validate(doc) for doc in documents]


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
    query_embedding = get_embedding(search.message)

    # Use SQLAlchemy ORM with pgvector's cosine_distance method
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
        .limit(search.limit)
    )

    try:
        result = await db.execute(stmt)
        rows = result.all()

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


@app.post("/api/chat", response_model=ChatMessage)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint - frontend controls the full conversation structure.
    This endpoint simply passes through what the frontend sends to the LLM.
    """
    if not genai_client:
        raise HTTPException(status_code=500, detail="Google API key is not set")

    # Convert ConversationMessage format to Google API format
    # Google Generative AI uses "user" and "model" roles (not "assistant")
    contents = []
    for msg in request.messages:
        # Convert "assistant" role to "model" for Google API
        role = "model" if msg.role == "assistant" else msg.role
        contents.append({"role": role, "parts": [{"text": msg.content}]})

    # Build config with system instruction if provided
    config = {}
    if request.system_instruction:
        config["system_instruction"] = request.system_instruction

    # Generate response using Gemini - pass through exactly what frontend sent
    try:
        # If contents is a single item, use it directly; otherwise use the list
        # This matches the API format from the examples
        contents_param = contents[0] if len(contents) == 1 else contents

        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_param,
            config=config if config else None,
        )

        # Extract the response text
        response_text = response.text

        # Build the full prompt representation for display
        # This shows exactly what was sent to the LLM
        full_prompt_parts = []
        if request.system_instruction:
            full_prompt_parts.append(f"System: {request.system_instruction}")
        for msg in request.messages:
            full_prompt_parts.append(f"{msg.role.title()}: {msg.content}")
        full_prompt = "\n\n".join(full_prompt_parts)

        # Note: retrieved_docs is None since we're stateless
        # Frontend should handle document retrieval separately if needed
        return ChatMessage(
            role="assistant",
            content=response_text,
            retrieved_docs=None,  # Frontend controls this
            full_prompt=full_prompt,
        )
    except genai_errors.ClientError as e:
        # Handle Google API client errors (quota, authentication, etc.)
        error_str = str(e)

        # Check if this is a quota/rate limit error (429)
        if hasattr(e, "status_code") and e.status_code == 429:
            # Extract retry delay if available
            retry_delay = None
            if "retry in" in error_str.lower():
                import re

                match = re.search(r"retry in ([\d.]+)s", error_str.lower())
                if match:
                    retry_delay = float(match.group(1))

            error_message = "API quota exceeded. You've reached the free tier limit (20 requests per day)."
            if retry_delay:
                error_message += f" Please retry in {int(retry_delay)} seconds."
            else:
                error_message += " Please try again later or check your billing plan."

            print(f"QUOTA ERROR in /api/chat: {error_str}")
            raise HTTPException(status_code=429, detail=error_message)

        # For other client errors, return appropriate status code
        status_code = getattr(e, "status_code", 500)
        error_details = f"API error: {error_str}"
        print(f"API ERROR in /api/chat: {error_details}")
        raise HTTPException(status_code=status_code, detail=error_details)

    except Exception as e:
        import traceback

        # For other errors, return 500 with detailed message
        error_details = f"Error generating response: {str(e)}"
        print(f"ERROR in /api/chat: {error_details}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_details)


@app.post("/api/conversation", response_model=ConversationResponse)
async def get_conversation(request: ConversationRequest):
    """
    Get the complete conversation structure that was sent to the LLM.
    Shows the retrieval query, retrieved chunks, and how the system instruction was augmented.
    """
    conversation_messages: List[ConversationMessage] = []

    # Add system message if provided (this is sent as system_instruction in config)
    if request.system_instruction:
        conversation_messages.append(
            ConversationMessage(role="system", content=request.system_instruction)
        )

    # Add all messages as-is (frontend already formatted them correctly)
    conversation_messages.extend(request.messages)

    # Build augmentation description if retrieval was performed
    augmentation = None
    if request.retrieval_query and request.retrieved_documents:
        base_instruction = request.base_system_instruction or "No base instruction"
        retrieved_content = "\n\n".join(
            [
                f"[Document {i+1} (similarity: {doc.similarity:.3f})]\n{doc.content}"
                for i, doc in enumerate(request.retrieved_documents)
            ]
        )
        augmentation = f"""Base System Instruction:
{base_instruction}

Retrieved Context (from query: "{request.retrieval_query}"):
{retrieved_content}

Augmented System Instruction:
{request.system_instruction}"""

    return ConversationResponse(
        messages=conversation_messages,
        retrieval_query=request.retrieval_query,
        retrieved_documents=request.retrieved_documents,
        base_system_instruction=request.base_system_instruction,
        augmentation=augmentation,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
