from sqlalchemy import Column, Integer, String
from app.db.database import Base

class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    # TODO: add relationships/constraints
