# Lecture 3: FastAPI + Svelte SPA

This project demonstrates a FastAPI backend serving a SvelteKit Single Page Application (SPA).

## Project Structure

```
lec3/
├── backend/
│   └── main.py          # FastAPI application
├── frontend/            # SvelteKit application
│   ├── src/
│   │   └── routes/
│   │       ├── +page.svelte
│   │       └── +layout.ts
│   └── package.json
└── docker-compose.yml   # PostgreSQL database (optional)
```

## Setup Instructions

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

This will install all required dependencies including `@sveltejs/adapter-static` for building the SPA.

### 2. Build the Frontend

```bash
cd frontend
npm run build
```

This creates a static build in `frontend/build/` that FastAPI will serve.

### 3. Install Backend Dependencies

From the project root (where `pyproject.toml` is located):

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 4. Run the Backend

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
- Health check: http://localhost:8000/api/health

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

- `GET /api/items` - Returns all items
- `POST /api/items` - Creates a new item
- `GET /api/health` - Health check endpoint

## Features

- ✅ FastAPI backend with CORS enabled
- ✅ SvelteKit SPA served by FastAPI
- ✅ Static file serving for assets
- ✅ SPA routing support (all routes serve index.html)
- ✅ Sample data display with modern UI
- ✅ Error handling and loading states
