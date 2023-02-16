from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Settings
from predictor import Predictor
from utils import bytes_to_ndarray

settings = Settings()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor(model_path=settings.model_path,
                      overlap_threshold=settings.overlap_threshold, confidence_threshold=settings.confidence_threshold)


class BoxResult(BaseModel):
    box: tuple[int, int, int, int]
    confidence: float


class PredictionResult(BaseModel):
    count: int
    message: str
    boxes: list[BoxResult]


@app.get("/")
async def root():
    return {"message": "Hello! I'm Buuz App."}


@app.post("/predict")
async def predict(file: UploadFile) -> PredictionResult:
    if file.content_type != "image/jpeg" and file.content_type != "image/png":
        raise HTTPException(
            status_code=415, detail="Only JPEG or PNG images are supported")
    if file.size is not None and file.size > settings.file_size_limit:
        raise HTTPException(status_code=413, detail="File size too large")
    image = bytes_to_ndarray(await file.read())
    boxes, pad_horizontal, pad_vertical = predictor.predict(image)
    objects = []
    for box in boxes:
        x1, y1, x2, y2, cnf, cls = box
        objects.append(BoxResult(box=(x1-pad_horizontal, y1-pad_vertical,
                       x2-pad_horizontal, y2-pad_vertical), confidence=cnf))
    return PredictionResult(count=len(objects), message=f"Ok", boxes=objects)
