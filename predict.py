"""
MODULO: predict.py
DESCRIZIONE: Esegue l'inferenza con la U-Net sui pesi migliori salvati
             e visualizza il confronto (Input, Maschera Reale, Predizione)
             su più campioni di test contemporaneamente.
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

from model import UNet

# Rilevamento automatico dell'acceleratore hardware:
# 1. DirectML (utilizzato nello sviluppo locale su GPU AMD Radeon)
# 2. CUDA (per GPU Nvidia standard)
# 3. CPU (fallback universale per massima portabilità ed esecuzione cross-platform)
try:
    import torch_directml
    device = torch_directml.device()
    device_name = torch_directml.device_name(0)
except Exception:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"


def pulisci_maschera(mask_binaria, kernel_size=5):
    """
    Isola la componente connessa principale, esegue flood-fill per riempire i buchi interni
    e applica una chiusura morfologica leggera.
    """
    mask_uint8 = (mask_binaria * 255).astype(np.uint8)

    # 1. Componente connessa principale
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    maschera_pulita = np.zeros_like(mask_uint8)
    if num_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        maschera_pulita[labels == max_label] = 255

    # 2. Flood Fill per eliminare buchi interni
    h, w = maschera_pulita.shape
    im_floodfill = maschera_pulita.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask_ff, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    maschera_piena = maschera_pulita | im_floodfill_inv

    # 3. Chiusura morfologica leggera
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    maschera_finale = cv2.morphologyEx(maschera_piena, cv2.MORPH_CLOSE, kernel)

    return (maschera_finale > 127).astype(np.float32)


def predict_benchmark(num_samples=4, img_size=(256, 256)):
    print(f"Dispositivo in uso per inferenza: {device_name}")

    path_custom = "outputs/unet_pesi.pth"
    path_tl = "outputs/unet_resnet34_pesi.pth"

    if not os.path.exists(path_custom) or not os.path.exists(path_tl):
        print("Errore: uno o entrambi i file dei pesi (.pth) non sono stati trovati in outputs/")
        return

    # 1. Carica modello Custom U-Net
    model_custom = UNet(in_channels=3, out_channels=1).to(device)
    model_custom.load_state_dict(torch.load(path_custom, map_location=device))
    model_custom.eval()

    # 2. Carica modello Transfer Learning (ResNet34)
    model_tl = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model_tl.load_state_dict(torch.load(path_tl, map_location=device))
    model_tl.eval()

    # 3. Selezione campioni di test
    all_files = sorted(os.listdir("data/images"))
    selected_files = all_files[-num_samples:]

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    print(f"Generazione confronto visivo per {num_samples} campioni...")

    with torch.no_grad():
        for i, file_name in enumerate(selected_files):
            img_path = os.path.join("data/images", file_name)
            mask_path = os.path.join("data/masks", file_name)

            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            img_resized = TF.resize(img_pil, img_size)
            mask_resized = TF.resize(mask_pil, img_size, interpolation=TF.InterpolationMode.NEAREST)

            # Normalizzazione coerente con il training
            img_t = TF.to_tensor(img_resized)
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_tensor = img_t.unsqueeze(0).to(device)

            # Inferenza Custom U-Net
            logits_c = model_custom(input_tensor)
            probs_c = torch.sigmoid(logits_c)
            pred_c = (probs_c > 0.5).float().squeeze().cpu().numpy()
            pred_c_clean = pulisci_maschera(pred_c)

            # Inferenza U-Net + ResNet34
            logits_tl = model_tl(input_tensor)
            probs_tl = torch.sigmoid(logits_tl)
            pred_tl = (probs_tl > 0.5).float().squeeze().cpu().numpy()
            pred_tl_clean = pulisci_maschera(pred_tl)

            # 1: Immagine originale
            axes[i, 0].imshow(img_resized)
            axes[i, 0].set_title(f"Originale [{file_name}]")
            axes[i, 0].axis("off")

            # 2: Ground Truth
            axes[i, 1].imshow(mask_resized, cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            # 3: Custom U-Net (From scratch)
            axes[i, 2].imshow(pred_c_clean, cmap="gray")
            axes[i, 2].set_title("Custom U-Net (Scratch)")
            axes[i, 2].axis("off")

            # 4: U-Net + ResNet34 (Transfer Learning)
            axes[i, 3].imshow(pred_tl_clean, cmap="gray")
            axes[i, 3].set_title("U-Net + ResNet34 (TL)")
            axes[i, 3].axis("off")

    plt.tight_layout()
    output_plot = "outputs/confronto_benchmark.png"
    plt.savefig(output_plot, dpi=160)
    plt.close()
    print(f"Confronto salvato con successo in: {output_plot}")


if __name__ == "__main__":
    predict_benchmark(num_samples=4)