"""
MODULO: dataset.py
DESCRIZIONE: Caricamento dati con Data Augmentation sincronizzata 
             tra immagine RGB e maschera binaria.
"""
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def _apply_augmentation(self, image, mask):
        # 1. Flip orizzontale casuale (50% probabilita')
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 2. Rotazione leggera sincronizzata (-15 a +15 gradi)
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Variazione fotometrica (SOLO immagine, MAI la maschera)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Ridimensionamento preliminare
        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        # Applicazione dell'augmentation se abilitata (in training)
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Conversione a tensori PyTorch e Normalizzazione
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
# --- VERIFICA DEL FUNZIONAMENTO ---
if __name__ == "__main__":
    dataset_aug = SegmentationDataset("data/images", "data/masks", augment=True)
    img_aug, mask_aug = dataset_aug[0]
    print(f"Dataset con Augmentation attivo. Campione estratto: forma img {img_aug.shape}, maschera {mask_aug.shape}")