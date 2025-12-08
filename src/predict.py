import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # dossier src/
MODELS_DIR = os.path.join(BASE_DIR, "../models")
MODELS_DIR = os.path.normpath(MODELS_DIR)
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

# ---------- Définition des architectures (pour reconstruire les modèles) ----------

def build_mobilenet_v2(num_classes: int):
    """
    Reconstruit MobileNetV2 avec la même tête que pendant l'entraînement.
    Attention: weights=None car on va charger NOS poids ensuite.
    """
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def build_resnet18(num_classes: int):
    """
    Reconstruit ResNet18 avec la même tête que pendant l'entraînement.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ---------- Classe prédicteur ----------

class WastePredictor:
    def __init__(self, arch: str = "mobilenetv2"):
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
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"--- IA de détection prête ({self.arch}) ---")

    # ---------- Gestion des chemins ----------

    def get_model_path(self):
        """
        Renvoie le chemin du fichier de poids en fonction de l'architecture.
        Adapté aux noms que tu utilises dans le notebook.
        """
        if self.arch == "mobilenetv2":
            filename = "mobilenetv2_trash.pth"   # <--- adapté
        elif self.arch == "resnet18":
            filename = "resnet18_trash.pth"      # <--- adapté
        else:
            raise ValueError(f"Architecture inconnue : {self.arch}")

        return os.path.join(MODELS_DIR, filename)

    # ---------- Chargement classes & modèle ----------

    def load_classes(self):
        if not os.path.exists(CLASSES_PATH):
            print(f"ERREUR: Fichier classes.json introuvable ({CLASSES_PATH})")
            return []
        with open(CLASSES_PATH, "r") as f:
            classes = json.load(f)
        print("Classes chargées :", classes)
        return classes

    def load_model(self):
        model_path = self.get_model_path()

        if not os.path.exists(model_path):
            print(f"ERREUR: Modèle introuvable ({model_path}). As-tu bien sauvegardé le .pth depuis le notebook ?")
            return None
            
        num_classes = len(self.classes)
        if num_classes == 0:
            print("ERREUR: aucune classe chargée, impossible de construire le modèle.")
            return None

        # Reconstruire l'architecture
        if self.arch == "mobilenetv2":
            model = build_mobilenet_v2(num_classes)
        elif self.arch == "resnet18":
            model = build_resnet18(num_classes)
        else:
            raise ValueError(f"Architecture inconnue : {self.arch}")
        
        # Charger les poids
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()  # Mode évaluation
        print(f"Poids chargés depuis : {model_path}")
        return model

    # ---------- Prédiction ----------

    def predict(self, image_or_path):
        """
        Prédit la classe d'une image (soit un chemin, soit une image PIL directe)
        Retourne (classe_prédite, probabilité_max)
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

# Instance par défaut (MobileNetV2)
predictor = WastePredictor(arch="mobilenetv2")
