from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Importa la configuración de la DB y el modelo
from database import SessionLocal, init_db, Prediction

# Inicializa la base de datos (crea la tabla 'predictions' si no existe)
init_db()

app = FastAPI()

# --- Esquemas Pydantic (Para la validación de datos) ---

# Esquema para crear un nuevo registro (lo que el usuario enviará)
class PredictionCreate(BaseModel):
    input_data: str
    prediction_result: str
    confidence: Optional[float] = None

# Esquema para leer un registro (lo que la API devolverá)
class PredictionSchema(BaseModel):
    id: int
    timestamp: datetime
    input_data: str
    prediction_result: str
    confidence: Optional[float]

    class Config:
        orm_mode = True # Habilita la compatibilidad con SQLAlchemy

# --- Dependencia de la Base de Datos ---

# Función para obtener una sesión de DB y cerrarla después de usarla
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Endpoints de la API ---

@app.get("/")
def read_root():
    return {"message": "Backend de Historial de Predicciones en funcionamiento."}

@app.post("/predictions/", response_model=PredictionSchema, status_code=201)
def create_prediction(prediction: PredictionCreate, db: Session = Depends(get_db)):
    """
    Guarda una nueva predicción en el historial.
    """
    db_prediction = Prediction(
        input_data=prediction.input_data,
        prediction_result=prediction.prediction_result,
        confidence=prediction.confidence
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

@app.get("/predictions/", response_model=List[PredictionSchema])
def read_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Obtiene el historial de predicciones.
    """
    predictions = db.query(Prediction).offset(skip).limit(limit).all()
    return predictions

@app.get("/predictions/{prediction_id}", response_model=PredictionSchema)
def read_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    """
    Obtiene una predicción específica por su ID.
    """
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if prediction is None:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
