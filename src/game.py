import pygame
from settings import * 
from robot import Robot       
from objects import Waste     
from environment import Environment 
from predict import WastePredictor   # ⬅️ on importe la classe, plus le predictor global

# ===================== PALETTE & UI =====================

UI_GREEN   = (0, 180, 120)   # Vert modern
UI_RED     = (230, 70, 70)   # Rouge vif
UI_ORANGE  = (245, 170, 60)  # Orange pour le doute
UI_BLACK   = (15, 20, 30)
UI_WHITE   = (245, 245, 245)
UI_GRAY    = (160, 170, 185)
UI_DARK_BG = (8, 12, 20)     # Fond HUD sombre
UI_PANEL   = (20, 30, 50)    # Panneaux HUD
UI_ACCENT  = (80, 200, 180)  # Accent pour barres

# Objectif "symbolique" de CO₂ pour la barre de progression
MAX_CO2_GOAL_KG = 5.0

# ===================== CONFIG CO2 PAR CLASSE =====================

CO2_CONFIG = {
    "Plastic": {
        "avg_weight_kg": 0.09,
        "co2_per_kg": 2.3,
    },
    "Metal": {
        "avg_weight_kg": 0.1,
        "co2_per_kg": 2.1,
    },
    "Glass": {
        "avg_weight_kg": 0.15,
        "co2_per_kg": 0.5,
    },
    "Paper": {
        "avg_weight_kg": 0.05,
        "co2_per_kg": 0.4,
    },
    "Natural": {
        "avg_weight_kg": 0.2,
        "co2_per_kg": 0.0,
    },
}


def compute_co2_for_item(label: str) -> float:
    cfg = CO2_CONFIG.get(label)
    if cfg is None:
        return 0.0
    return cfg["avg_weight_kg"] * cfg["co2_per_kg"]


def draw_vertical_gradient(screen, top_color, bottom_color):
    h = SCREEN_HEIGHT
    for y in range(h):
        ratio = y / h
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))


def draw_button(screen, rect, text, font, active=False, hint=None):
    x, y, w, h = rect
    radius = 18

    base_color = (30, 40, 70) if not active else (50, 80, 130)
    border_color = (120, 190, 255) if active else (80, 100, 150)

    pygame.draw.rect(screen, border_color, rect, border_radius=radius)
    inner_rect = pygame.Rect(x + 2, y + 2, w - 4, h - 4)
    pygame.draw.rect(screen, base_color, inner_rect, border_radius=radius)

    text_surf = font.render(text, True, UI_WHITE)
    text_rect = text_surf.get_rect(center=(x + w // 2, y + h // 2 - 8))
    screen.blit(text_surf, text_rect)

    if hint:
        hint_font = pygame.font.Font(None, 24)
        hint_surf = hint_font.render(hint, True, UI_GRAY)
        hint_rect = hint_surf.get_rect(center=(x + w // 2, y + h // 2 + 18))
        screen.blit(hint_surf, hint_rect)


def choose_model(screen, clock):
    font_title = pygame.font.Font(None, 64)
    font_sub   = pygame.font.Font(None, 30)
    font_btn   = pygame.font.Font(None, 36)
    font_hint  = pygame.font.Font(None, 26)

    choosing = True
    arch = "mobilenetv2"  # valeur par défaut

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

        draw_vertical_gradient(screen, (10, 20, 40), (5, 10, 20))

        header = pygame.Surface((SCREEN_WIDTH, 120), pygame.SRCALPHA)
        header.fill((0, 0, 0, 80))
        screen.blit(header, (0, 0))

        title_surf = font_title.render("Green AI – Model Selection", True, UI_WHITE)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, 45))
        screen.blit(title_surf, title_rect)

        sub_surf = font_sub.render("Choose your vision model to power the robot:", True, UI_GRAY)
        sub_rect = sub_surf.get_rect(center=(SCREEN_WIDTH // 2, 90))
        screen.blit(sub_surf, sub_rect)

        card_width = 380
        card_height = 140
        spacing = 40
        total_width = card_width * 2 + spacing

        start_x = (SCREEN_WIDTH - total_width) // 2
        y = SCREEN_HEIGHT // 2 - card_height // 2

        rect1 = (start_x, y, card_width, card_height)
        draw_button(
            screen,
            rect1,
            "1  •  MobileNetV2",
            font_btn,
            active=True,
            hint="Fast & energy-efficient"
        )

        rect2 = (start_x + card_width + spacing, y, card_width, card_height)
        draw_button(
            screen,
            rect2,
            "2  •  ResNet18",
            font_btn,
            active=False,
            hint="Deeper, potentially more accurate"
        )

        hint_text = "Press 1 or 2 to start the simulation"
        hint_surf = font_hint.render(hint_text, True, UI_GRAY)
        hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(hint_surf, hint_rect)

        pygame.display.flip()
        clock.tick(60)

    print(f"[MENU] Modèle sélectionné : {arch}")
    predictor = WastePredictor(arch=arch)
    return predictor


def draw_confidence_bar(screen, x, y, width, height, value):
    pygame.draw.rect(screen, UI_GRAY, (x, y, width, height), border_radius=8)
    fill_width = int(width * max(0.0, min(1.0, value)))
    if fill_width > 0:
        pygame.draw.rect(screen, UI_ACCENT, (x, y, fill_width, height), border_radius=8)


def draw_co2_bar(screen, x, y, width, height, current_kg):
    pygame.draw.rect(screen, UI_GRAY, (x, y, width, height), border_radius=8)
    ratio = max(0.0, min(1.0, current_kg / MAX_CO2_GOAL_KG))
    fill_width = int(width * ratio)
    if fill_width > 0:
        pygame.draw.rect(screen, UI_GREEN, (x, y, fill_width, height), border_radius=8)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Green AI - Analyse Performance & Confiance")
    clock = pygame.time.Clock()
    
    # Fonts
    font_main    = pygame.font.Font(None, 40)  # pour gros chiffres (100.0%)
    font_status  = pygame.font.Font(None, 34)  # pour "Prediction: CORRECT"
    font_sub     = pygame.font.Font(None, 30)
    font_label   = pygame.font.Font(None, 24)
    font_stats   = pygame.font.Font(None, 26)
    font_small   = pygame.font.Font(None, 20)  # un peu plus petit pour gagner de la place

    predictor = choose_model(screen, clock)

    env = Environment()  
    robot = Robot()      
    
    all_sprites = pygame.sprite.Group(robot)
    waste_group = pygame.sprite.Group()

    last_real = "..."
    last_pred = "..."
    last_conf = 0.0

    collected_counts = {cls: 0 for cls in CO2_CONFIG.keys()}
    total_co2_kg = 0.0

    SPAWN_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_EVENT, 1500) 

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == SPAWN_EVENT:
                if len(waste_group) < 6:
                    w = Waste()
                    all_sprites.add(w)
                    waste_group.add(w)

        all_sprites.update()

        hits = pygame.sprite.spritecollide(robot, waste_group, True)
        for w in hits:
            if w.image_path:
                pred, conf = predictor.predict(w.image_path)
                last_real = w.category
                last_pred = pred
                last_conf = conf

                label = w.category
                if label in collected_counts:
                    collected_counts[label] += 1
                    gained_kg = compute_co2_for_item(label)
                    total_co2_kg += gained_kg
                    print(f"[CO2] Ramassé {label} → +{gained_kg:.3f} kg CO₂ évitée (total={total_co2_kg:.3f} kg)")
                
                print(f"[Analyse] Réel: {last_real} | IA: {last_pred} ({conf:.0%})")

        env.draw(screen)
        all_sprites.draw(screen)

        for waste in waste_group:
            text_surf = font_label.render(waste.category, True, BLUE)
            text_rect = text_surf.get_rect(center=(waste.rect.centerx, waste.rect.y - 15))
            screen.blit(text_surf, text_rect)

        # === HUD EN BAS ===
        hud_height = 200        # 🔧 un peu plus haut
        hud_y = SCREEN_HEIGHT - hud_height

        hud_bg = pygame.Surface((SCREEN_WIDTH, hud_height), pygame.SRCALPHA)
        hud_bg.fill((5, 10, 20, 220))
        screen.blit(hud_bg, (0, hud_y))

        column_width = SCREEN_WIDTH // 3
        padding = 12
        panel_radius = 20

        panel1_rect = pygame.Rect(
            padding,
            hud_y + padding,
            column_width - 2 * padding,
            hud_height - 2 * padding
        )
        panel2_rect = pygame.Rect(
            column_width + padding,
            hud_y + padding,
            column_width - 2 * padding,
            hud_height - 2 * padding
        )
        panel3_rect = pygame.Rect(
            2 * column_width + padding,
            hud_y + padding,
            column_width - 2 * padding,
            hud_height - 2 * padding
        )

        pygame.draw.rect(screen, UI_PANEL, panel1_rect, border_radius=panel_radius)
        pygame.draw.rect(screen, UI_PANEL, panel2_rect, border_radius=panel_radius)
        pygame.draw.rect(screen, UI_PANEL, panel3_rect, border_radius=panel_radius)

        # ---------- PANEL 1 : ACCURACY ----------
        if last_pred == "...":
            acc_color = UI_GRAY
            status_label = "Waiting for first prediction..."
            icon = "•"
        elif last_real == last_pred:
            acc_color = UI_GREEN
            status_label = "Prediction: CORRECT"
            icon = "✓"
        else:
            acc_color = UI_RED
            status_label = "Prediction: WRONG"
            icon = "✗"

        p1_title = font_small.render("MODEL ACCURACY", True, UI_GRAY)
        screen.blit(p1_title, (panel1_rect.x + 16, panel1_rect.y + 12))

        # 🔧 police réduite pour éviter le débordement
        status_surf = font_status.render(f"{icon} {status_label}", True, acc_color)
        screen.blit(status_surf, (panel1_rect.x + 16, panel1_rect.y + 40))

        if last_pred != "...":
            detail_real = font_small.render(f"Ground truth : {last_real}", True, UI_WHITE)
            detail_pred = font_small.render(f"Model output : {last_pred}", True, UI_WHITE)
            screen.blit(detail_real, (panel1_rect.x + 16, panel1_rect.y + 85))
            screen.blit(detail_pred, (panel1_rect.x + 16, panel1_rect.y + 110))

        # ---------- PANEL 2 : CONFIANCE ----------
        p2_title = font_small.render("MODEL CONFIDENCE", True, UI_GRAY)
        screen.blit(p2_title, (panel2_rect.x + 16, panel2_rect.y + 12))

        if last_conf > 0.75:
            conf_color = UI_GREEN
            conf_label = "Confident"
        elif last_conf > 0.0:
            conf_color = UI_ORANGE
            conf_label = "Uncertain"
        else:
            conf_color = UI_GRAY
            conf_label = "N/A"

        conf_text = f"{last_conf:.1%}" if last_pred != "..." else "--"
        conf_surf = font_main.render(conf_text, True, conf_color)
        conf_rect = conf_surf.get_rect()
        conf_rect.midleft = (panel2_rect.x + 20, panel2_rect.y + 70)
        screen.blit(conf_surf, conf_rect)

        conf_label_surf = font_small.render(conf_label, True, conf_color)
        conf_label_rect = conf_label_surf.get_rect()
        conf_label_rect.topleft = (panel2_rect.x + 20, panel2_rect.y + 105)
        screen.blit(conf_label_surf, conf_label_rect)

        bar_x = panel2_rect.x + 16
        bar_y = panel2_rect.y + 135
        bar_w = panel2_rect.width - 32
        bar_h = 18
        draw_confidence_bar(screen, bar_x, bar_y, bar_w, bar_h, last_conf if last_pred != "..." else 0.0)

        # ---------- PANEL 3 : CO₂ & COMPTEURS ----------
        p3_title = font_small.render("ECO IMPACT", True, UI_GRAY)
        screen.blit(p3_title, (panel3_rect.x + 16, panel3_rect.y + 12))

        co2_text = font_stats.render(f"CO2 avoided: {total_co2_kg:.2f} kg", True, UI_WHITE)
        screen.blit(co2_text, (panel3_rect.x + 16, panel3_rect.y + 35))

        co2_bar_x = panel3_rect.x + 16
        co2_bar_y = panel3_rect.y + 60
        co2_bar_w = panel3_rect.width - 32
        co2_bar_h = 18
        draw_co2_bar(screen, co2_bar_x, co2_bar_y, co2_bar_w, co2_bar_h, total_co2_kg)

        goal_text = font_small.render(
            f"Symbolic goal: {MAX_CO2_GOAL_KG:.1f} kg",
            True,
            UI_GRAY
        )
        screen.blit(goal_text, (panel3_rect.x + 16, panel3_rect.y + 86))

        # 🔧 point de départ + espacement ajustés pour que les 5 lignes tiennent
        start_y_counts = panel3_rect.y + 104
        line_spacing = 17
        for i, (cls, count) in enumerate(collected_counts.items()):
            line = f"{cls}: {count}"
            txt_surf = font_small.render(line, True, UI_WHITE)
            screen.blit(txt_surf, (panel3_rect.x + 16, start_y_counts + i * line_spacing))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
