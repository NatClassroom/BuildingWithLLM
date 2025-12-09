"""Database models"""

from sqlalchemy import Column, Integer, String
from backend.database import Base


class ItemModel(Base):
    """SQLAlchemy model for items table"""

    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
