from io import BytesIO
from PIL import Image
import numpy as np


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))
