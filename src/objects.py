import pygame
import os
import random
from settings import DATASET_SAND_DIR, SCREEN_WIDTH, SCREEN_HEIGHT, RED

class Waste(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        
        # Gestion d'erreur si le dataset n'existe pas
        if not os.path.exists(DATASET_SAND_DIR):
            self.image = pygame.Surface((30, 30))
            self.image.fill(RED)
            self.image_path = None
            self.category = "Unknown"
            return

        # Choix aléatoire
        self.category = random.choice(os.listdir(DATASET_SAND_DIR))
        cat_path = os.path.join(DATASET_SAND_DIR, self.category)
        
        valid_images = [f for f in os.listdir(cat_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        
        if not valid_images:
            self.image = pygame.Surface((30, 30))
            self.image.fill(RED)
            self.image_path = None
        else:
            img_name = random.choice(valid_images)
            self.image_path = os.path.join(cat_path, img_name)
            
            loaded_img = pygame.image.load(self.image_path).convert()
            self.image = pygame.transform.scale(loaded_img, (60, 60))

        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - 60)
        self.rect.y = random.randrange(0, SCREEN_HEIGHT - 60)