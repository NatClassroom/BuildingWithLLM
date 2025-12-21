# Lecture 5: RAG (Retrieval-Augmented Generation) Demo

This project demonstrates a complete RAG (Retrieval-Augmented Generation) process with a visual, step-by-step interface.

## Features

- **File Upload & Processing**: Upload a text file and see it processed through:
  - Text extraction
  - Text chunking
  - Vector embedding generation
  - Vector database storage
- **RAG Chat Interface**: Chat with the system and see:
  - Conversation messages
  - Document retrieval process
  - Retrieved context augmentation
  - AI-generated responses using retrieved context

## Project Structure

```
lec5/
├── backend/
│   ├── main.py          # FastAPI application with RAG endpoints
│   ├── models.py        # SQLAlchemy models with pgvector support
│   └── database.py      # Database connection with pgvector setup
├── frontend/            # SvelteKit application
│   ├── src/
│   │   └── routes/
│   │       └── +page.svelte  # RAG demo UI with tabs
│   └── package.json
├── alembic/             # Database migrations
│   └── versions/
│       └── 001_create_documents_table.py
└── docker-compose.yml   # PostgreSQL with pgvector extension
```

## Setup Instructions

### 1. Set Up Environment Variables

Create a `.env` file in the project root (or set environment variable):

```bash
GEMINI_API_KEY=your_google_api_key_here
```

### 2. Start the Database

Start the PostgreSQL database with pgvector extension using Docker Compose:

```bash
cd lec5
docker-compose up -d
```

This will start a PostgreSQL database with pgvector extension on port 5434.

### 3. Install Backend Dependencies

From the project root (where `pyproject.toml` is located):

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 4. Run Database Migrations

Run Alembic migrations to set up the database schema:

```bash
cd lec5
alembic upgrade head
```

This will:

- Enable the pgvector extension
- Create the `documents` table with vector embedding column
- Create an index for efficient vector similarity search

### 5. Start the Backend Server

From the `lec5` directory:

```bash
python -m backend.main
```

Or using uvicorn directly:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 6. Start the Frontend

From the `lec5/frontend` directory:

```bash
cd frontend
npm install  # First time only
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

### Upload Tab

1. Click on the "📄 Upload & Process" tab
2. Select a `.txt` file to upload
3. Watch the step-by-step process:
   - **Text Extraction**: See the extracted text from your file
   - **Text Chunking**: See how the text is split into chunks
   - **Vector Embedding**: See embeddings generated for each chunk
   - **Vector Database Storage**: See documents stored in the database

### Chat Tab

1. Click on the "💬 Chat with RAG" tab
2. Type a question about the uploaded documents
3. See the RAG process in action:
   - Your message appears in the conversation
   - The system retrieves relevant document chunks
   - Retrieved chunks are shown with similarity scores
   - The AI generates a response using the retrieved context

## API Endpoints

- `POST /api/upload` - Upload a text file and extract/chunk it
- `POST /api/embed` - Generate embeddings for text chunks
- `POST /api/documents` - Store chunks with embeddings in vector database
- `GET /api/documents` - Retrieve all documents
- `POST /api/documents/search` - Search for similar documents
- `POST /api/chat` - Chat with RAG (retrieval + generation)

## How RAG Works

1. **Retrieval**: When you ask a question, the system:

   - Generates an embedding for your query
   - Searches the vector database for similar document chunks
   - Returns the most relevant chunks based on cosine similarity

2. **Augmentation**: The retrieved chunks are:

   - Combined into a context string
   - Added to the prompt sent to the AI model

3. **Generation**: The AI model:
   - Receives your question + retrieved context
   - Generates an answer based on both

This allows the AI to answer questions about your specific documents, even though it wasn't trained on them!
