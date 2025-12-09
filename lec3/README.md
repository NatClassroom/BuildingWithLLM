# Lecture 3: FastAPI + Svelte SPA

This project demonstrates a FastAPI backend serving a SvelteKit Single Page Application (SPA).

## Project Structure

```
lec3/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── models.py        # SQLAlchemy database models
│   └── database.py     # Database connection and session management
├── frontend/            # SvelteKit application
│   ├── src/
│   │   └── routes/
│   │       ├── +page.svelte
│   │       └── +layout.ts
│   └── package.json
├── alembic/             # Database migrations
│   ├── versions/        # Migration files
│   └── env.py           # Alembic environment configuration
├── alembic.ini          # Alembic configuration
└── docker-compose.yml   # PostgreSQL database
```

## Setup Instructions

### 1. Start the Database

Start the PostgreSQL database using Docker Compose:

```bash
docker-compose up -d
```

This will start a PostgreSQL database with pgvector extension on port 5432.

### 2. Install Backend Dependencies

From the project root (where `pyproject.toml` is located):

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Run Database Migrations

Run Alembic migrations to set up the database schema:

```bash
alembic upgrade head
```

This will create the necessary database tables (e.g., `items` table).

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

This will install all required dependencies including `@sveltejs/adapter-static` for building the SPA.

### 5. Build the Frontend

```bash
cd frontend
npm run build
```

This creates a static build in `frontend/build/` that FastAPI will serve.

### 6. Run the Backend

```bash
cd backend
python main.py
```

Or using uvicorn directly:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:

- Frontend: http://localhost:8000
- API: http://localhost:8000/api/items

## Development Workflow

### Frontend Development

For frontend development with hot reload:

```bash
cd frontend
npm run dev
```

This runs the SvelteKit dev server (usually on port 5173). You'll need to configure the API proxy or use CORS.

### Backend Development

The backend runs on port 8000. After making changes to `backend/main.py`, the server will auto-reload if you use the `--reload` flag.

### Production Build

1. Build the frontend: `cd frontend && npm run build`
2. Run the backend: `cd backend && python main.py`

FastAPI will automatically serve the built SPA from `frontend/build/`.

## API Endpoints

- `GET /api/items` - Returns all items with a message and count
  - Response: `{ "message": "Hello from the API!", "items": [...], "count": N }`
- `POST /api/items` - Creates a new item
  - Request body: `{ "name": string, "description": string }`
  - Response: `{ "id": int, "name": string, "description": string }`

## Features

- ✅ FastAPI backend with CORS enabled
- ✅ SvelteKit SPA served by FastAPI
- ✅ Static file serving for assets
- ✅ SPA routing support (all routes serve index.html)
- ✅ Sample data display with modern UI
- ✅ Error handling and loading states
