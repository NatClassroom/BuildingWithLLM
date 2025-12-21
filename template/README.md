# Lecture 4: Vector Database Demo

This project demonstrates vector database retrieval techniques using pgvector with PostgreSQL.

## Features

- **Document Storage**: Add documents that are automatically embedded using Google Generative AI embeddings
- **Semantic Search**: Search for similar documents using cosine similarity on vector embeddings
- **Vector Retrieval**: Demonstrates how vector databases enable semantic search beyond keyword matching

## Project Structure

```
lec4/
├── backend/
│   ├── main.py          # FastAPI application with vector search endpoints
│   ├── models.py        # SQLAlchemy models with pgvector support
│   └── database.py      # Database connection with pgvector setup
├── frontend/            # SvelteKit application
│   ├── src/
│   │   └── routes/
│   │       └── +page.svelte  # Vector search demo UI
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
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Start the Database

Start the PostgreSQL database with pgvector extension using Docker Compose:

```bash
cd lec4
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
cd lec4
alembic upgrade head
```

This will:

- Enable the pgvector extension
- Create the `documents` table with vector embedding column
- Create an index for efficient vector similarity search

### 5. Install Frontend Dependencies

```bash
cd lec4/frontend
npm install
```

### 6. Build the Frontend

```bash
cd lec4/frontend
npm run build
```

This creates a static build in `frontend/build/` that FastAPI will serve.

### 7. Run the Backend

```bash
cd lec4/backend
python main.py
```

Or using uvicorn directly:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:

- Frontend: http://localhost:8000
- API: http://localhost:8000/api/documents

## How It Works

### Vector Embeddings

1. When you add a document, the backend:

   - Generates a 768-dimensional embedding using Google's `text-embedding-004` model
   - Stores both the document content and its embedding vector in PostgreSQL

2. When you search:
   - Your query is converted to an embedding vector
   - The database finds documents with similar embeddings using cosine distance
   - Results are ranked by similarity score (0-100%)

### Vector Similarity Search

The demo uses **cosine similarity** to find semantically similar documents:

- Cosine similarity measures the angle between vectors
- Documents with similar meaning have similar embeddings
- This enables semantic search beyond exact keyword matching

### Example Usage

1. **Add Documents**:

   - "Python is a high-level programming language"
   - "Machine learning uses algorithms to learn from data"
   - "FastAPI is a modern web framework for Python"

2. **Search**:
   - Query: "programming" → Finds the Python document
   - Query: "AI algorithms" → Finds the machine learning document
   - Query: "web development" → Finds the FastAPI document

## API Endpoints

- `POST /api/documents` - Add a new document (auto-generates embedding)
- `GET /api/documents` - Get all documents
- `POST /api/documents/search` - Search for similar documents
  ```json
  {
    "query": "your search query",
    "limit": 5
  }
  ```

## Technologies Used

- **pgvector**: PostgreSQL extension for vector similarity search
- **Google Generative AI**: For generating text embeddings
- **FastAPI**: Backend API framework
- **SQLAlchemy**: ORM with asyncpg driver
- **SvelteKit**: Frontend framework
