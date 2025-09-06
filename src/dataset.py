from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class PartsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []  # (path, part_id)
        for part_dir in self.root.iterdir():
            if not part_dir.is_dir(): 
                continue
            pid = part_dir.name
            for img in part_dir.iterdir():
                if img.suffix.lower() in [".jpg",".jpeg",".png"]:
                    self.samples.append((img, pid))
        self.samples.sort()

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, pid = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, pid
