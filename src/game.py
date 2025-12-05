import pygame
from settings import * 
from robot import Robot       
from objects import Waste     
from environment import Environment 
from predict import WastePredictor   # ⬅️ on importe la classe, plus le predictor global

# Couleurs Spécifiques pour l'UI
UI_GREEN = (0, 140, 0)   # Vert foncé lisible
UI_RED = (220, 0, 0)     # Rouge vif
UI_ORANGE = (230, 120, 0)# Orange pour le doute
UI_BLACK = (0, 0, 0)

def choose_model(screen, clock):
    """
    Petit écran de sélection du modèle au lancement du jeu.
    Retourne une instance de WastePredictor initialisée avec le bon modèle.
    """
    font_title = pygame.font.Font(None, 60)
    font_option = pygame.font.Font(None, 40)
    font_hint = pygame.font.Font(None, 28)

    choosing = True
    arch = "mobilenetv2"  # valeur par défaut si jamais

    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    arch = "mobilenetv2"
                    choosing = False
                elif event.key == pygame.K_2:
                    arch = "resnet18"
                    choosing = False

        # Fond
        screen.fill((240, 240, 240))

        # Textes
        title_surf = font_title.render("Choisissez le modèle IA", True, UI_BLACK)
        opt1_surf = font_option.render("1 - MobileNetV2 (rapide, léger)", True, UI_BLACK)
        opt2_surf = font_option.render("2 - ResNet18 (plus profond)", True, UI_BLACK)
        hint_surf = font_hint.render("Appuie sur 1 ou 2 pour commencer", True, UI_BLACK)

        # Placement
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
        opt1_rect = opt1_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
        opt2_rect = opt2_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
        hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 90))

        screen.blit(title_surf, title_rect)
        screen.blit(opt1_surf, opt1_rect)
        screen.blit(opt2_surf, opt2_rect)
        screen.blit(hint_surf, hint_rect)

        pygame.display.flip()
        clock.tick(30)

    # Quand le joueur a choisi, on crée le bon prédicteur
    print(f"[MENU] Modèle sélectionné : {arch}")
    predictor = WastePredictor(arch=arch)
    return predictor


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Green AI - Analyse Performance & Confiance")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.Font(None, 40)  # Gros pour le résultat
    font_sub = pygame.font.Font(None, 30)   # Moyen pour la confiance
    font_label = pygame.font.Font(None, 24) # Petit pour les objets

    # --- CHOIX DU MODÈLE AVANT TOUT ---
    predictor = choose_model(screen, clock)

    # --- INITIALISATION DU JEU ---
    env = Environment()  
    robot = Robot()      
    
    all_sprites = pygame.sprite.Group(robot)
    waste_group = pygame.sprite.Group()

    # Variables pour stocker la dernière analyse
    last_real = "..."      # La vraie catégorie (Réalité)
    last_pred = "..."      # Ce que l'IA a vu (Prédiction)
    last_conf = 0.0        # Le pourcentage de certitude

    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, 1500) 

    running = True
    while running:
        # 1. Événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == SPAWN_EVENT:
                if len(waste_group) < 6:
                    w = Waste()
                    all_sprites.add(w)
                    waste_group.add(w)

        # 2. Mise à jour
        all_sprites.update()

        # 3. INTERACTION ROBOT / DÉCHETS
        hits = pygame.sprite.spritecollide(robot, waste_group, True)
        for w in hits:
            if w.image_path:
                # L'IA fait sa prédiction avec le modèle choisi
                pred, conf = predictor.predict(w.image_path)
                
                # On sauvegarde tout pour l'affichage
                last_real = w.category
                last_pred = pred
                last_conf = conf
                
                # Petit log console pour ton rapport
                print(f"[Analyse] Réel: {last_real} | IA: {last_pred} ({conf:.0%})")

        # 4. Affichage
        env.draw(screen)
        all_sprites.draw(screen)

        # Labels bleus au-dessus des objets (La Réalité)
        for waste in waste_group:
            text_surf = font_label.render(waste.category, True, BLUE)
            text_rect = text_surf.get_rect(center=(waste.rect.centerx, waste.rect.y - 15))
            screen.blit(text_surf, text_rect)

        # --- INTERFACE INTELLIGENTE (UI) ---
        # Fond blanc semi-transparent (plus grand pour 2 lignes)
        ui_bg = pygame.Surface((SCREEN_WIDTH, 90))
        ui_bg.set_alpha(230)
        ui_bg.fill(WHITE)
        screen.blit(ui_bg, (0, 0))

        # --- LIGNE 1 : PRÉCISION (Est-ce que c'est bon ?) ---
        if last_pred == "...":
            acc_color = UI_BLACK
            acc_text = "En attente d'analyse..."
        elif last_real == last_pred:
            acc_color = UI_GREEN
            acc_text = f"CORRECT ! (Réel: {last_real} = IA: {last_pred})"
        else:
            acc_color = UI_RED
            acc_text = f"ERREUR ! (Réel: {last_real} ≠ IA: {last_pred})"

        screen.blit(font_main.render(acc_text, True, acc_color), (20, 10))

        # --- LIGNE 2 : CONFIANCE (Est-ce qu'elle doute ?) ---
        if last_conf > 0.75:  # Seuil de confiance haute
            conf_color = UI_GREEN
            conf_label = "Confiante"
        elif last_conf > 0.0:
            conf_color = UI_ORANGE  # Orange pour le doute
            conf_label = "Hésitante"
        else:
            conf_color = UI_BLACK
            conf_label = "..."

        if last_pred != "...":
            conf_text = f"Niveau de Confiance : {last_conf:.1%} ({conf_label})"
            screen.blit(font_sub.render(conf_text, True, conf_color), (20, 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
