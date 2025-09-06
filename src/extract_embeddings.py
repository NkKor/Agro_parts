import argparse
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import joblib
from PIL import Image

from config import TARGET_SIZE
from utils_cv import load_image_folder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # убираем последний слой
    model.eval()
    return model.to(DEVICE)


def extract_embedding(model, img_path):
    transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(x).cpu().numpy().flatten()
    return emb


def run_embedding(src_root: Path, dst_root: Path):
    model = build_model()
    embeddings = {}
    for class_dir in tqdm(list(src_root.iterdir()), desc="extract"):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        embs = []
        for img_file in class_dir.glob("*.jpg"):
            try:
                emb = extract_embedding(model, img_file)
                embs.append(emb)
            except Exception as e:
                print(f"⚠️ {img_file}: {e}")
        if embs:
            embeddings[class_id] = np.vstack(embs)
    dst_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(embeddings, dst_root / "embeddings.pkl")
    print(f"✅ Сохранено в {dst_root / 'embeddings.pkl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Path to processed images")
    parser.add_argument("--dst", type=str, help="Path to embeddings folder")
    args = parser.parse_args()

    if args.src and args.dst:
        run_embedding(Path(args.src), Path(args.dst))
    else:
        print("⚠️ Запуск без аргументов: используем processed/ → embeddings/")
        run_embedding(Path("processed"), Path("embeddings"))


if __name__ == "__main__":
    main()
