from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from models import loader
from services import inference, gradcam
from schemas.requests import PredictRequest, GradCAMRequest

os.makedirs("static", exist_ok=True)
app = FastAPI(title="MNIST CNN Visual Analytics API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    loader.load()

@app.post("/predict")
async def predict(request: PredictRequest):
    if loader.activation_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return inference.run_predict(request.image)

@app.post("/gradcam")
async def gradcam_endpoint(request: GradCAMRequest):
    if loader.gradcam_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return gradcam.run_gradcam(request.image, request.class_index)

@app.post("/get_kernel_weights")
async def get_kernel_weights(payload: dict):
    layer_name = payload.get("layer_name")
    kernel_idx = int(payload.get("kernel_idx"))
    
    # Hand off execution to your clean service layer function
    result = inference.run_get_kernel_weights(layer_name, kernel_idx)
    return result

@app.get("/data.json")
async def get_data():
    if not os.path.exists("data.json"):
        raise HTTPException(status_code=404, detail="Run train_and_extract.py first.")
    return FileResponse("data.json")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools():
    return {}

app.mount("/", StaticFiles(directory="static", html=True), name="static")