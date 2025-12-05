import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from codecarbon import EmissionsTracker 

# Modules locaux
from dataset_loader import get_data_loaders
from model_definition import build_model, build_model_resnet  # <-- important

# --- PARAMÈTRES GLOBAUX ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

def train_one_model(arch, DATA_DIR, MODELS_DIR):
    """
    Entraîne UN modèle (MobileNetV2 ou ResNet) selon la fonction appropriée.
    """

    print(f"\n==============================")
    print(f"   TRAINING MODEL: {arch}")
    print(f"==============================")

    # Chargement dataset
    train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    if train_loader is None:
        return

    # Tracker CO2 spécifique à chaque modèle
    tracker = EmissionsTracker(
        output_dir=MODELS_DIR,
        output_file=f"emissions_{arch}.csv",
        project_name=f"GreenAI_Waste_{arch}",
        on_csv_write="update"
    )
    tracker.start()

    # Sélection du constructeur de modèle
    if arch == "mobilenetv2":
        model = build_model(len(class_names), pretrained=True).to(DEVICE)
    elif arch == "resnet18":
        model = build_model_resnet(len(class_names), pretrained=True).to(DEVICE)
    else:
        raise ValueError(f"Architecture inconnue : {arch}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Boucle d'entraînement
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{arch}] Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

        tracker.flush()

    # Sauvegarde du modèle
    save_path = os.path.join(MODELS_DIR, f"best_{arch}_sand.pth")
    torch.save(model.state_dict(), save_path)

    tracker.stop()

    print(f"\n[✓] Modèle sauvegardé : {save_path}")
    print(f"[✓] Fichier CO2 : emissions_{arch}.csv")


def train():
    # Chemins
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "dataset_sand")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print(f"Training on device: {DEVICE}")

    # Entraîner les deux modèles 🎯
    for arch in ["mobilenetv2", "resnet18"]:
        train_one_model(arch, DATA_DIR, MODELS_DIR)


if __name__ == "__main__":
    train()
