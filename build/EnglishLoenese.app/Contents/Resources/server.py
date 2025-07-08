from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import Dict
from PIL import Image
import onnxruntime as ort
from scipy.special import log_softmax

app = FastAPI()

ort_session = ort.InferenceSession('crnn_model.onnx', providers=['CPUExecutionProvider'])
charset = " !\"'(),./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert('L')
    img = img.resize((128, 32))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # channel
    img = np.expand_dims(img, axis=0)  # batch
    return img

def ctc_greedy_decoder(output, charset):
    pred = output.argmax(axis=2).squeeze(axis=0)
    prev = -1
    decoded = []
    for p in pred:
        if p != prev and p != 0:
            decoded.append(charset[p - 1])
        prev = p
    return ''.join(decoded)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    image_bytes = await file.read()
    img_array = preprocess_image_bytes(image_bytes)
    inputs = {ort_session.get_inputs()[0].name: img_array}
    outputs = ort_session.run(None, inputs)[0]
    outputs = log_softmax(outputs, axis=2)
    pred_text = ctc_greedy_decoder(outputs, charset)
    return {"result": pred_text}
