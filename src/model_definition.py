import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes, pretrained=True):
    print("Construction du modèle MobileNetV2...")
    
    # 1. Télécharger l'architecture MobileNetV2
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # 2. "Geler" le cerveau (Transfer Learning)
    # On ne touche pas aux connaissances de base (vision des formes/couleurs)
    # Cela accélère énormément l'entraînement.
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Remplacer la dernière couche pour tes 5 classes
    # MobileNet range sa sortie dans 'classifier[1]'
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model




def build_model_resnet(num_classes, arch="mobilenetv2", pretrained=True):
    print(f"Construction du modèle {arch} (pretrained={pretrained})")

    if arch == "mobilenetv2":
        # 1. Charger MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        # 2. Geler les features
        for param in model.features.parameters():
            param.requires_grad = False

        # 3. Remplacer la dernière couche
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        # 1. Charger ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # 2. Geler tout le backbone (sauf la fc)
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

        # 3. Remplacer la couche fully-connected
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Architecture inconnue : {arch}")

    return model
