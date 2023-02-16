from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Settings
from predictor import Predictor
from utils import bytes_to_ndarray

settings = Settings()

app = FastAPI()

origins = [
    "https://buuz.app",
    "https://api.buuz.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor(model_path=settings.model_path,
                      overlap_threshold=settings.overlap_threshold, confidence_threshold=settings.confidence_threshold)


class BoxResult(BaseModel):
    box: tuple[float, float, float, float]
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
    boxes, w, h, pad_horizontal, pad_vertical = predictor.predict(image)
    objects = []
    for box in boxes:
        x1, y1, x2, y2, cnf, cls = box
        objects.append(BoxResult(box=((x1-pad_horizontal) / w, (y1-pad_vertical) / h,
                       (x2-pad_horizontal) / w, (y2-pad_vertical) / h), confidence=cnf))
    return PredictionResult(count=len(objects), message=f"Ok", boxes=objects)
