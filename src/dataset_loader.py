import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, val_split=0.2):
    print(f"Chargement des données depuis : {data_dir}")

    # 1. Prétraitement (Standard pour MobileNet/ResNet)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Taille standard
        transforms.ToTensor(),          # Convertir en chiffres
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Charger les images
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    except FileNotFoundError:
        print(f"ERREUR: Le dossier {data_dir} n'existe pas. Lance augment_with_sand.py d'abord !")
        return None, None, None

    class_names = full_dataset.classes
    print(f"Classes trouvées : {class_names}")

    # 3. Sauvegarder la liste des classes (pour le jeu plus tard)
    # On remonte d'un cran pour aller dans 'models'
    models_dir = os.path.join(os.path.dirname(data_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    json_path = os.path.join(models_dir, 'classes.json')
    with open(json_path, 'w') as f:
        json.dump(class_names, f)
        print(f"Fichier classes.json sauvegardé ici : {json_path}")

    # 4. Séparer Entraînement (80%) / Validation (20%)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 5. Créer les chargeurs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names