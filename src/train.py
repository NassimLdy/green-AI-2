import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from codecarbon import EmissionsTracker 

# Modules locaux
from dataset_loader import get_data_loaders
from model_definition import build_model

def train():
    # --- CONFIGURATION DES CHEMINS ---
    # Calcul des chemins absolus pour éviter les erreurs d'écriture
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Dossier src
    ROOT_DIR = os.path.dirname(BASE_DIR)                  # Dossier racine
    DATA_DIR = os.path.join(ROOT_DIR, "dataset_sand")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    
    # 1. SÉCURITÉ : Création du dossier models s'il n'existe pas
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Dossier créé : {MODELS_DIR}")

    # 2. CONFIGURATION DU TRACKER CO2
    # Le tracker va écrire automatiquement dans models/emissions.csv
    tracker = EmissionsTracker(
        output_dir=MODELS_DIR, 
        output_file="emissions.csv",
        project_name="GreenAI_Waste",
        on_csv_write="update"
    )
    
    tracker.start()

    # --- PARAMÈTRES D'ENTRAÎNEMENT ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5 # Suffisant pour générer le rapport CO2 rapidement

    print(f"Démarrage entraînement sur {DEVICE}...")
    
    # Chargement des données
    train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    if train_loader is None: return

    # Chargement du modèle
    model = build_model(len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    # Boucle d'apprentissage
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} terminée.")
        
        # Force l'écriture des données intermédiaires
        tracker.flush()

    # --- FINALISATION ---
    save_path = os.path.join(MODELS_DIR, "best_model_sand.pth")
    torch.save(model.state_dict(), save_path)
    
    # Arrêt du tracker : c'est ici que le fichier final s'écrit
    tracker.stop()
    
    print("-" * 30)
    print("Entraînement terminé.")
    print(f"Modèle sauvegardé : {save_path}")
    
    # Vérification de l'existence du fichier CSV
    csv_path = os.path.join(MODELS_DIR, "emissions.csv")
    if os.path.exists(csv_path):
        print(f"SUCCÈS : Le fichier {csv_path} a été généré automatiquement.")
        print("Vous pouvez maintenant lancer la cellule du graphique dans le Notebook.")
    else:
        print(f"ERREUR : Le fichier {csv_path} n'a pas été trouvé.")

if __name__ == "__main__":
    train()