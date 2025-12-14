"""Database models"""

from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from backend.database import Base


class DocumentModel(Base):
    """SQLAlchemy model for documents with vector embeddings"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(
        Vector(768), nullable=True
    )  # 768 dimensions for common embedding models
