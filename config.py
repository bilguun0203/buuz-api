import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "./assets/model.onnx")
    confidence_threshold: float = os.getenv("CONFIDENCE_THRESHOLD", 0.5)
    overlap_threshold: float = os.getenv("OVERLAP_THRESHOLD", 0.5)
    file_size_limit: int = os.getenv("FILE_SIZE_LIMIT", 10485760)
    
    class Config:
        env_file = ".env"