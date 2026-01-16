import torch
from model import RadarFusionWeightNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/20260116_1615/best.pt", map_location=device)
print("best epoch:", ckpt["epoch"])
print("val loss at best:", ckpt["val_loss"])