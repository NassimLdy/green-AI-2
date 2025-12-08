import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, GREEN
import os
import cv2
import numpy as np
import os


class Robot(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        # Chemin vers l'image du robot
        asset_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        robot_path = os.path.join(asset_dir, "robot_sand.png")

        # Charger l'image
        self.image = pygame.image.load(robot_path).convert_alpha()

        # Optionnel : redimensionner si trop grand
        self.image = pygame.transform.scale(self.image, (60, 60))

        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        self.speed = 5

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_UP]:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN]:
            self.rect.y += self.speed

        # Garder le robot dans l'écran
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
