from sqlalchemy import Column, Integer, String
from app.db.database import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    # TODO: add relationships/constraints
