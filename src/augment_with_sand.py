import os
from PIL import Image
from rembg import remove
from tqdm import tqdm  # Barre de progression

# --- CONFIGURATION DES CHEMINS ---
# On se base sur l'emplacement de ce script (dossier 'src')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins relatifs vers les assets et le dataset
SAND_IMG_PATH = os.path.join(BASE_DIR, "../assets/sand/sand1.jpg")
RAW_DATASET_DIR = os.path.join(BASE_DIR, "../dataset/RealWaste")
OUTPUT_DIR = os.path.join(BASE_DIR, "../dataset_sand")

# Taille cible pour le modèle (ResNet utilise souvent 224x224)
TARGET_SIZE = (224, 224)

def process_dataset():
    print("--- Démarrage de la génération du dataset 'Sand' ---")

    # 1. Charger l'image de fond (Sable)
    if not os.path.exists(SAND_IMG_PATH):
        print(f"ERREUR CRITIQUE : Image de sable introuvable ici -> {SAND_IMG_PATH}")
        return

    try:
        sand_bg_original = Image.open(SAND_IMG_PATH).convert("RGBA")
        # On prépare le fond à la bonne taille tout de suite
        sand_bg_resized = sand_bg_original.resize(TARGET_SIZE)
    except Exception as e:
        print(f"Erreur lors du chargement du sable : {e}")
        return

    # 2. Créer le dossier de sortie
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier créé : {OUTPUT_DIR}")

    # 3. Lister les catégories (Glass, Metal, etc.)
    if not os.path.exists(RAW_DATASET_DIR):
        print(f"ERREUR : Dossier dataset introuvable -> {RAW_DATASET_DIR}")
        return

    categories = [d for d in os.listdir(RAW_DATASET_DIR) if os.path.isdir(os.path.join(RAW_DATASET_DIR, d))]
    
    print(f"Catégories détectées : {categories}")

    # 4. Traitement des images
    for category in categories:
        src_folder = os.path.join(RAW_DATASET_DIR, category)
        dst_folder = os.path.join(OUTPUT_DIR, category)
        
        # Créer le sous-dossier de destination (ex: dataset_sand/Glass)
        os.makedirs(dst_folder, exist_ok=True)

        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nTraitement de la classe : {category} ({len(images)} images)")

        # Boucle sur chaque image avec barre de progression
        for img_name in tqdm(images, desc=f"Processing {category}"):
            try:
                img_path = os.path.join(src_folder, img_name)
                
                # A. Ouvrir l'image du déchet
                obj_img = Image.open(img_path).convert("RGBA")

                # B. Supprimer le fond (magie de rembg)
                obj_no_bg = remove(obj_img)

                # C. Redimensionner l'objet pour qu'il rentre dans le carré de sable
                # On le réduit un peu (ex: 180px) pour qu'il ne touche pas les bords
                obj_no_bg.thumbnail((180, 180)) 

                # D. Préparer une copie propre du fond de sable
                final_comp = sand_bg_resized.copy()

                # E. Centrer l'objet
                bg_w, bg_h = final_comp.size
                obj_w, obj_h = obj_no_bg.size
                offset = ((bg_w - obj_w) // 2, (bg_h - obj_h) // 2)

                # F. Coller l'objet sur le sable
                # Le 3ème argument est le masque (pour gérer la transparence)
                final_comp.paste(obj_no_bg, offset, obj_no_bg)

                # G. Sauvegarder en JPG (RGB, sans transparence)
                final_path = os.path.join(dst_folder, img_name)
                # Convertir en RGB car le JPG ne supporte pas la transparence
                final_comp.convert("RGB").save(final_path, quality=95)

            except Exception as e:
                # Si une image est corrompue, on l'affiche mais on continue
                print(f" -> Erreur sur {img_name} : {e}")

    print("\n--- TERMINE ! ---")
    print(f"Votre nouveau dataset est prêt dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()