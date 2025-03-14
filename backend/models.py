from sqlalchemy import Column,Integer,String
from db import Base

class Translations(Base):
    __tablename__ = "Translations"
    id = Column(Integer, primary_key=True, index=True)
    english = Column(String(100), nullable=False)
    bengali = Column(String(100), nullable=False)