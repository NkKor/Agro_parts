import argparse, matplotlib.pyplot as plt, numpy as np, cv2 as cv
from pathlib import Path
from search_prod import search
from config import PROC_DIR

def show_results(query, results, per_pid_example=1):
    # показываем запрос и по 1 примеру из processed/ для каждого part_id
    cols = min(5, len(results)+1)
    rows = int(np.ceil((len(results)+1)/cols))
    plt.figure(figsize=(3.2*cols, 3.2*rows))

    # 1) запрос
    q = cv.imread(query)[:,:,::-1]
    plt.subplot(rows, cols, 1)
    plt.imshow(q); plt.title("QUERY"); plt.axis('off')

    # 2) кандидаты
    for i,(pid,score) in enumerate(results, start=2):
        part_dir = PROC_DIR/str(pid)
        ex = next(iter(p for p in part_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]), None)
        if ex is None: 
            img = np.ones((256,256,3), dtype=np.uint8)*220
        else:
            img = cv.imread(str(ex))[:,:,::-1]
        plt.subplot(rows, cols, i)
        plt.imshow(img); plt.axis('off'); plt.title(f"{pid}\n{score:.3f}")

    plt.tight_layout(); plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, type=str)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    res = search(args.query, topk=args.topk, rerank_per_image=True)
    show_results(args.query, res)

if __name__ == "__main__":
    main()
