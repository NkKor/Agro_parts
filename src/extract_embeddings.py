import numpy as np, torch, torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset import PartsDataset
from encoder import ResNet50Encoder
from config import (PROC_DIR, EMB_DIR, BATCH_SIZE, NUM_WORKERS, DEVICE)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # ImageNet
])

def main():
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ds = PartsDataset(str(PROC_DIR), transform=transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ResNet50Encoder().to(DEVICE).eval()
    all_emb, all_ids = [], []

    with torch.no_grad():
        for imgs, pids in tqdm(dl, desc="extract"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            emb = model(imgs).cpu().numpy()       # [B, D], L2-нормированы
            all_emb.append(emb)
            all_ids.extend(list(pids))

    feats = np.vstack(all_emb)                    # [N, D]
    np.save(EMB_DIR/"per_image.npy", feats)
    np.save(EMB_DIR/"part_ids.npy", np.array(all_ids, dtype=object))
    print("✅ saved:", EMB_DIR/"per_image.npy", EMB_DIR/"part_ids.npy")

if __name__ == "__main__":
    main()
