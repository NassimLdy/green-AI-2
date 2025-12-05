import pygame
import os
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, ASSETS_DIR

class Environment:
    def __init__(self):
        # On définit le chemin vers l'image du sable
        self.sand_path = os.path.join(ASSETS_DIR, "sand/sand1.jpg")
        self.background = self.load_background()

    def load_background(self):
        """Charge l'image et la met à la bonne taille, ou met du gris si erreur."""
        try:
            bg = pygame.image.load(self.sand_path).convert()
            bg = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
            return bg
        except FileNotFoundError:
            print(f"Attention: Image de fond introuvable ({self.sand_path})")
            # Fallback : un fond gris simple
            bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            bg.fill((200, 200, 200))
            return bg

    def draw(self, screen):
        """Affiche le fond sur l'écran"""
        screen.blit(self.background, (0, 0))