from sqlalchemy import Column, Integer, String
from app.db.database import Base

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    # TODO: add relationships/constraints
