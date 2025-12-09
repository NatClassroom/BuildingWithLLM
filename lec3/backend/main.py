from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Building with LLM",
    description="Tutorial project for Building with LLM",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/data")
async def get_data():
    """Simple GET endpoint that returns sample data"""
    return {
        "message": "Hello from the API!",
        "items": [
            {"id": 1, "name": "Item 1", "description": "This is the first item. Yay"},
            {"id": 2, "name": "Item 2", "description": "This is the second item"},
            {"id": 3, "name": "Item 3", "description": "This is the third item"},
            {"id": 4, "name": "Item 4", "description": "This is the fourth item"},
        ],
        "count": 4,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
