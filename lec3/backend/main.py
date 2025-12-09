import sys
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    backend_dir = Path(__file__).parent
    parent_dir = backend_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db
from backend.models import ItemModel

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


# Pydantic models
class ItemCreate(BaseModel):
    """Request model for creating an item"""

    name: str
    description: str


class Item(BaseModel):
    """Response model for an item"""

    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


@app.get("/api/items")
async def get_data(db: AsyncSession = Depends(get_db)):
    """Simple GET endpoint that returns sample data"""
    result = await db.execute(select(ItemModel))
    items = result.scalars().all()

    return {
        "message": "Hello from the API!",
        "items": [Item.model_validate(item).model_dump() for item in items],
        "count": len(items),
    }


@app.post("/api/items", response_model=Item)
async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    """POST endpoint to create a new item"""
    new_item = ItemModel(name=item.name, description=item.description)
    db.add(new_item)
    await db.commit()
    await db.refresh(new_item)

    return Item.model_validate(new_item)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
