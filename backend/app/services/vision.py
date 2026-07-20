import io
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

class CNNVisionService:
    def __init__(self, model_path: str = "app/data/models/crop_disease_cnn.onnx"):
        # 1. Fallback path check so it works locally AND inside Docker
        if not os.path.exists(model_path):
            alt_path = "data/models/crop_disease_cnn.onnx"
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"❌ ONNX model not found at '{model_path}' or '{alt_path}'!")

        print(f"✅ Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 29 Class names matching your EfficientNet training output exactly
        self.classes = [
            "Apple__Apple_Scab","Apple__Black_Rot","Apple__Cedar_Apple_Rust","Apple__Healthy",
            "Bell_Pepper__Bacterial_Spot","Bell_Pepper__Healthy","Cherry__Healthy","Cherry__Powder_Mildew",
            "Corn_(Maize)__Cercospora_Leaf_Spot","Corn_(Maize)__Common_Rust","Corn_(Maize)__Healthy",
            "Corn_(Maize)__Northern_Leaf_Blight","Grape__Black_Rot","Grape__Esca_(Black_Measles)",
            "Grape__Healthy","Grape__Leaf_Blight","Peach__Bacterial_Spot","Peach__Healthy",
            "Potato__Early_Blight","Potato__Healthy","Potato__Late_Blight","Strawberry__Healthy", 
            "Strawberry__Leaf_Scorch","Tomato__Bacterial_Spot","Tomato__Early_Blight","Tomato__Healthy",
            "Tomato__Late_Blight","Tomato__Septoria_Leaf_Spot","Tomato__Yellow_Leaf_Curl_Virus"
        ]

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Preprocesses raw image bytes into a 4D normalized NumPy tensor for CNN input."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        
        img_data = np.array(image, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data

    def predict(self, image_bytes: bytes) -> dict:
        """Runs CNN inference and returns prediction metadata."""
        input_tensor = self._preprocess(image_bytes)
        
        raw_logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        probabilities = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]
        
        predicted_idx = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_idx])
        predicted_class = self.classes[predicted_idx] if predicted_idx < len(self.classes) else "Unknown"
        
        # 2. FIXED: Split on TWO underscores to match self.classes!
        if "__" in predicted_class:
            crop_name, disease = predicted_class.split("__", 1)
        else:
            crop_name, disease = "Unknown", predicted_class
        
        clean_disease_name = disease.replace("_", " ")
        clean_crop_name = crop_name.replace("_", " ")
        
        return {
            "disease_id": clean_disease_name,
            "crop_name": clean_crop_name,
            "confidence": confidence_score,
            "initial_summary": f"Detected {clean_disease_name} on {clean_crop_name} with {confidence_score * 100:.1f}% confidence."
        }