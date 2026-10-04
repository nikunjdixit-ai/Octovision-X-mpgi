from sqlalchemy import Column, Integer, String
from app.db.database import Base

class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    # TODO: add relationships/constraints
