from pydantic import BaseSettings


class Settings(BaseSettings):
    model_path: str = "./assets/model.onnx"
    confidence_threshold: float = 0.5
    overlap_threshold: float = 0.5
    file_size_limit: int = 10485760
    
    class Config:
        env_file = ".env"