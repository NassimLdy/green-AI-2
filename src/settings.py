import os

# --- ÉCRAN ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# --- COULEURS ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# --- CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "../assets")
DATASET_SAND_DIR = os.path.join(BASE_DIR, "../dataset_sand")