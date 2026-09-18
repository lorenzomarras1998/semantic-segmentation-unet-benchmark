"""
MODULO: download_data.py
DESCRIZIONE: Scarica un dataset reale tramite il catalogo ufficiale di Torchvision,
             estraendo immagini reali e maschere binarie in data/images e data/masks.
"""

import os
import shutil
import numpy as np
from PIL import Image
import torchvision.datasets as datasets

def setup_torchvision_dataset():
    print("--- Download Dataset Reale tramite Torchvision ---")
    raw_dir = "temp_raw_data"
    
    # Download del benchmark di segmentazione ufficiale
    dataset = datasets.OxfordIIITPet(
        root=raw_dir,
        split="trainval",
        target_types="segmentation",
        download=True
    )
    
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)

    # Selezioniamo un sottoinsieme di 500 campioni per un test rapido e pulito
    num_samples = 2000
    print(f"Estrazione e preparazione di {num_samples} campioni reali...")

    for i in range(num_samples):
        img, trimap = dataset[i]
        
        # Nome file standard
        file_name = f"sample_{i:03d}.png"
        img_save_path = os.path.join("data/images", file_name)
        mask_save_path = os.path.join("data/masks", file_name)

        # 1. Salva immagine RGB
        img.save(img_save_path)

        # 2. Conversione della maschera in binaria (0: sfondo, 1: foreground)
        trimap_np = np.array(trimap)
        binary_mask = (trimap_np == 1).astype(np.uint8) * 255
        Image.fromarray(binary_mask).save(mask_save_path)

    # Pulizia cartella provvisoria
    shutil.rmtree(raw_dir)
    print(f"Operazione completata! 500 immagini e maschere reali salvate in data/images e data/masks.")

if __name__ == "__main__":
    setup_torchvision_dataset()
