import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json

# Import de vos architectures
from model_definition import build_model, build_model_resnet

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

class WastePredictor:
    def __init__(self, arch="mobilenetv2"):
        """
        arch : 'mobilenetv2' ou 'resnet18'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.classes = self.load_classes()
        self.model = self.load_model()
        
        # Les mêmes transformations que pour l'entraînement !
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        print(f"--- IA de détection prête ({self.arch}) ---")

    # ---------- Gestion des chemins ----------

    def get_model_path(self):
        """
        Renvoie le chemin du fichier de poids en fonction de l'architecture.
        À adapter si tu as choisi d'autres noms de fichiers.
        """
        if self.arch == "mobilenetv2":
            filename = "best_mobilenetv2_sand.pth"
        elif self.arch == "resnet18":
            filename = "best_resnet18_sand.pth"
        else:
            raise ValueError(f"Architecture inconnue : {self.arch}")

        return os.path.join(MODELS_DIR, filename)

    # ---------- Chargement classes & modèle ----------

    def load_classes(self):
        if not os.path.exists(CLASSES_PATH):
            print(f"ERREUR: Fichier classes.json introuvable ({CLASSES_PATH})")
            return []
        with open(CLASSES_PATH, 'r') as f:
            return json.load(f)

    def load_model(self):
        model_path = self.get_model_path()

        if not os.path.exists(model_path):
            print(f"ERREUR: Modèle introuvable ({model_path}). As-tu lancé train.py ?")
            return None
            
        # Reconstruit l'architecture correspondante
        if self.arch == "mobilenetv2":
            model = build_model(num_classes=len(self.classes), pretrained=False)
        elif self.arch == "resnet18":
            model = build_model_resnet(num_classes=len(self.classes), pretrained=False)
        else:
            raise ValueError(f"Architecture inconnue : {self.arch}")
        
        # Charge les poids
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()  # Mode évaluation
        return model

    # ---------- Prédiction ----------

    def predict(self, image_or_path):
        """
        Prédit la classe d'une image (soit un chemin, soit une image PIL directe)
        """
        if self.model is None:
            return "Erreur Modèle", 0.0

        # Gestion du format d'entrée (Chemin fichier ou Image PIL)
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        # Préparer l'image (Tenseur)
        input_tensor = self.transform(image).unsqueeze(0)  # (1, 3, 224, 224)
        input_tensor = input_tensor.to(self.device)

        # Prédiction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted_idx.item()]
        confidence = score.item()

        return predicted_class, confidence

# Instance par défaut (MobileNetV2) – utile si tu ne gères pas le choix utilisateur
predictor = WastePredictor(arch="mobilenetv2")
