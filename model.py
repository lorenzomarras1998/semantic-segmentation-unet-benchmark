"""
MODULO: model.py
DESCRIZIONE: Architettura U-Net modulare per segmentazione semantica binaria.
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Blocco base: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder (Discesa)
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
        
        # Bottleneck (Punto più profondo)
        self.bottleneck = DoubleConv(64, 128)
        
        # Decoder (Salita)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 64 -> 128
        self.dec2 = DoubleConv(128, 64)  # 64 (dal decoder) + 64 (dallo skip) = 128 in ingresso
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 128 -> 256
        self.dec1 = DoubleConv(64, 32)   # 32 (dal decoder) + 32 (dallo skip) = 64 in ingresso
        
        # Convoluzione 1x1 finale
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # 2. Bottleneck
        b = self.bottleneck(p2)
        
        # 3. Decoder con Skip Connections
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)  # Ponte dal livello e2
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Ponte dal livello e1
        d1 = self.dec1(d1)
        
        # 4. Mappa finale (logit grezzi)
        return self.final_conv(d1)


# --- TEST DI VERIFICA DIMENSIONI ---
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    
    # Simuliamo un batch di 2 immagini RGB 256x256
    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_output = model(dummy_input)
    
    print("========================================")
    print("Modello U-Net creato con successo!")
    print(f"Dimensioni Input:  {dummy_input.shape}  -> (Batch, Canali, H, W)")
    print(f"Dimensioni Output: {dummy_output.shape} -> (Batch, Canali, H, W)")
    print("========================================")