import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_image

class Predictor:
    def __init__(self, model_path='models/model.keras'):
        self.model_path = model_path
        self.model = None
        self.load()

    def load(self):
        """Loads the model into memory"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            print(f"Warning: Model not found at {self.model_path}. Predictions will fail.")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Predicts whether the image is a Cat or a Dog.
        Returns the class name and confidence score.
        """
        if self.model is None:
            self.load()
            if self.model is None:
                raise ValueError("Model is not loaded.")
        
        processed_image = preprocess_image(image_bytes)
        
        # Predict
        prediction_prob = self.model.predict(processed_image)[0][0]
        
        # Based on binary classification (typically 0=Cat, 1=Dog according to sorted folders)
        # Verify your exact mapping from ImageDataGenerator.class_indices
        label = "Dog" if prediction_prob > 0.5 else "Cat"
        confidence = float(prediction_prob) if label == "Dog" else 1.0 - float(prediction_prob)
        
        return {
            "prediction": label,
            "confidence": confidence,
            "raw_probability": float(prediction_prob)
        }

# Initialize a global predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor

def predict_image(image_bytes: bytes) -> dict:
    ped = get_predictor()
    return ped.predict(image_bytes)
