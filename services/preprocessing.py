import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image

def decode_image(b64_str: str) -> tf.Tensor:
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('L')
    arr = np.array(img.resize((28, 28))).astype('float32') / 255.0
    return tf.convert_to_tensor(arr.reshape(1, 28, 28, 1))

def array_to_base64(arr: np.ndarray) -> str:
    arr_min, arr_max = arr.min(), arr.max()
    normed = (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)
    img = Image.fromarray((normed * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()