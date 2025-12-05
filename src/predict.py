import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json

# Import de votre architecture MobileNet
from model_definition import build_model

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_sand.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

class WastePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.load_classes()
        self.model = self.load_model()
        
        # Les mêmes transformations que pour l'entraînement !
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        print("--- IA de détection prête ---")

    def load_classes(self):
        if not os.path.exists(CLASSES_PATH):
            print(f"ERREUR: Fichier classes.json introuvable ({CLASSES_PATH})")
            return []
        with open(CLASSES_PATH, 'r') as f:
            return json.load(f)

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"ERREUR: Modèle introuvable ({MODEL_PATH}). Avez-vous lancé train.py ?")
            return None
            
        # On reconstruit l'architecture MobileNet avec le bon nombre de classes
        model = build_model(num_classes=len(self.classes), pretrained=False)
        
        # On charge les poids entraînés
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.to(self.device)
        model.eval() # Mode évaluation (pas d'apprentissage)
        return model

    def predict(self, image_or_path):
        """
        Prédit la classe d'une image (soit un chemin, soit une image PIL directe)
        """
        if self.model is None:
            return "Erreur Modèle"

        # Gestion du format d'entrée (Chemin fichier ou Image PIL)
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        # Préparer l'image (Tenseur)
        input_tensor = self.transform(image).unsqueeze(0) # Ajouter dimension batch (1, 3, 224, 224)
        input_tensor = input_tensor.to(self.device)

        # Prédiction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted_idx.item()]
        confidence = score.item()

        return predicted_class, confidence

# Instance globale pour être appelée facilement depuis game.py
predictor = WastePredictor()