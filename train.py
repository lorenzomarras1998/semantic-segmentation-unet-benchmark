"""
MODULO: train.py
DESCRIZIONE: Pipeline di addestramento e validazione per la U-Net
             utilizzando la loss combinata BCEDiceLoss e checkpoint automatico.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset

from model import UNet
from dataset import SegmentationDataset
from metrics import dice_score, BCEDiceLoss

def train_unet():
    # 1. Configurazione hardware
    import torch_directml
    device = torch_directml.device()
    print(f"Dispositivo in uso: {torch_directml.device_name(0)}")

    # 2. Iperparametri
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 50
    train_ratio = 0.8

    # 3. Preparazione dataset e dataloaders
    dataset_train_raw = SegmentationDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=(256, 256),
        augment=True
    )
    dataset_val_raw = SegmentationDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=(256, 256),
        augment=False
    )

    total_samples = len(dataset_train_raw)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    # Generazione indici casuali riproducibili
    generator = torch.Generator().manual_seed(42)
    indici = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = indici[:train_size]
    val_indices = indici[train_size:]

    train_dataset = Subset(dataset_train_raw, train_indices)
    val_dataset = Subset(dataset_val_raw, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Campioni totali: {total_samples} (Train: {train_size}, Val: {val_size})")

    # 4. Inizializzazione modello, loss combinata e ottimizzatore
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/unet_pesi.pth"

    miglior_val_dice = 0.0

    print("--- Inizio Addestramento ---")
    for epoch in range(1, num_epochs + 1):
        # Fase di Addestramento
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_dice += dice_score(outputs, masks) * images.size(0)

        epoch_train_loss = train_loss / train_size
        epoch_train_dice = train_dice / train_size

        # Fase di Validazione
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                val_dice += dice_score(outputs, masks) * images.size(0)

        epoch_val_dice = val_dice / val_size

        print(
            f"Epoca [{epoch:02d}/{num_epochs:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Dice: {epoch_train_dice:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f}"
        )

        # Salvataggio del checkpoint ottimale
        if epoch_val_dice > miglior_val_dice:
            miglior_val_dice = epoch_val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  -> Checkpoint aggiornato salvato (Nuovo Record Val Dice: {miglior_val_dice:.4f})")

    print("----------------------------")
    print(f"Training terminato! Miglior Dice in validazione: {miglior_val_dice:.4f}")

if __name__ == "__main__":
    train_unet()