from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# URL de la base de datos SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"

# Crea el motor de la base de datos
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para las clases del modelo
Base = declarative_base()

# --- Modelo de Predicción ---
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(String)  # Los datos de entrada que alimentaron al modelo
    prediction_result = Column(String) # El resultado de la predicción (ej: "Clase A" o valor)
    confidence = Column(Float, nullable=True) # Nivel de confianza del modelo

# Función para crear la tabla
def init_db():
    Base.metadata.create_all(bind=engine)