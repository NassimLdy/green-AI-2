
# Projet — Module de Prédiction et Jeu Interactif

Ce projet combine un système de prédiction en temps réel et un jeu interactif. Pour que l'application fonctionne correctement, les scripts doivent être exécutés dans un ordre précis, et le modèle doit être sélectionné depuis l'interface avant de démarrer une partie.

## 1. Prérequis

- Python 3.8 ou version ultérieure


## 2. Lancer le système de prédiction

Le script `predict.py` doit être exécuté avant le lancement du jeu.

Ce script initialise le modèle de prédiction. Le jeu dépend de ce module pour récupérer et analyser les données en temps réel. Si `predict.py` n’est pas lancé, le jeu ne pourra pas fonctionner correctement.

## 3. Lancer le jeu

Une fois `predict.py` opérationnel, lancez le jeu avec :


## 4. Sélection du modèle dans l'interface

Dans la page d’accueil de l’interface, vous devez choisir quel modèle utiliser avant de commencer une partie.  
La sélection s'effectue via les raccourcis clavier suivants :

- Shift + 1 : sélection du Modèle 1  
- Shift + 2 : sélection du Modèle 2  

Sans cette sélection, la prédiction ne sera pas correctement attribuée au modèle souhaité.


