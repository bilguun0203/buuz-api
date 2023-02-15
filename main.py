from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI()

class PredictionResult(BaseModel):
    count: int
    message: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile) -> PredictionResult:
    if file.content_type != "image/jpeg" and file.content_type != "image/png":
        raise HTTPException(status_code=415, detail="Only JPEG or PNG images are supported")
    return PredictionResult(count=0, message="Ok")