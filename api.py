from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException
from fastapi.responses import JSONResponse
import time
import os
import shutil
from typing import List
from src.prediction import predict_image
from src.model import retrain_model

app = FastAPI(title="ML Pipeline API")

# Setup startup time for uptime calculation
START_TIME = time.time()

# Directories for training
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    """Returns the API health and uptime in seconds."""
    uptime = time.time() - START_TIME
    return {"status": "ok", "uptime_seconds": round(uptime, 2)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predicts Cat or Dog for a single uploaded image."""
    try:
        contents = await file.read()
        result = predict_image(contents)
        return {"filename": file.filename, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_data")
async def upload_bulk_data(label: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Uploads multiple images to be added to the training set for a specific label.
    `label` must be 'cats' or 'dogs'.
    """
    label = label.lower()
    if label not in ['cats', 'dogs']:
        raise HTTPException(status_code=400, detail="Label must be 'cats' or 'dogs'.")
    
    target_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(target_dir, exist_ok=True)
    
    saved_files = []
    for f in files:
        file_path = os.path.join(target_dir, f.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append(f.filename)

    return {"message": f"Successfully uploaded {len(saved_files)} files to '{label}' directory.", "files": saved_files}

def background_retrain_task():
    try:
        print("Starting background retraining task...")
        # Start retraining from current data directories
        retrain_model(train_dir=TRAIN_DIR, validation_dir=VAL_DIR)
        print("Retraining completed.")
    except Exception as e:
        print(f"Error during retraining: {e}")

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Triggers the model retraining process in the background."""
    background_tasks.add_task(background_retrain_task)
    return {"message": "Retraining task triggered in the background. It may take some time."}
