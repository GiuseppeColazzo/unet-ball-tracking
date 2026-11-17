import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
SIGMA = 2.0
EMPTY = np.zeros((0,2), dtype=np.float32)



def make_heatmap(centers, height, width, sigma):
    heatmap = torch.zeros(height, width)
    xs = torch.arange(width, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32) 

    # centers -> (N,2) -> cx, cy for each ball
    for cx, cy in centers:
        dx2 = ((xs + 0.5 - cx) ** 2) [None,:]
        dy2 = ((ys + 0.5 -cy) ** 2) [:, None]
        g   = torch.exp(-(dx2 + dy2) / (2 * sigma**2))
        heatmap = torch.maximum(heatmap, g)


    return heatmap.unsqueeze(0) # (1, H, W)


def load_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))              # BGR (H, W, 3) uint8, or None
    if frame is None:
        raise FileNotFoundError(f"file not found {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



class BallDataset(Dataset):
    def __init__(self, split= "train", root=PROCESSED, sigma=SIGMA, augment=False):
        super().__init__()

        self.samples = []
        self.root = root
        self.sigma = sigma
        self.augment = augment     # random flip + brightness/contrast, only for training
        meta = json.loads((root / "meta.json").read_text())
        self.width, self.height = meta["size"]
        self.centers = {}


        labels = pd.read_parquet(path= root / "labels.parquet", )
        labels = labels[labels["split"] ==split]
        if labels.empty:
            raise ValueError("Labels not found")

        last_frames = labels.groupby("clip")["frame"].max()
        for clip, last in last_frames.items():
            for t in range(2,last):
                self.samples.append((clip,t))

        vis = labels[labels["visible"]]

        for (clip, frame), g in vis.groupby(["clip","frame"]):
            self.centers[(clip,frame)] = g[["cx_s","cy_s"]].to_numpy(dtype=np.float32)



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, i):
        clip, t = self.samples[i]

        paths = []
        for frame in range(t-1, t+2):
            paths.append(self.root / "frames" / clip / f"{frame:06d}.jpg")

        imgs = [load_frame(p) for p in paths]

        x = np.concatenate(imgs, axis=2)
        x = torch.from_numpy(x).permute(2,0,1).contiguous().float() / 255
        centers = self.centers.get((clip,t), EMPTY)
        heatmap = make_heatmap(centers, self.height, self.width, self.sigma)

        if self.augment:
            # horizontal flip: frames, heatmap and ball positions together
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
                heatmap = torch.flip(heatmap, dims=[2])
                centers = centers.copy()                    # don't modify the arrays in self.centers
                centers[:, 0] = self.width - centers[:, 0]

            # same brightness and contrast change on all 3 frames
            brightness = 0.8 + 0.4 * torch.rand(1).item()   # between 0.8 and 1.2
            contrast = 0.8 + 0.4 * torch.rand(1).item()
            mean = x.mean()
            x = ((x - mean) * contrast + mean) * brightness
            x = x.clamp(0, 1)

        visible = len(centers) > 0

        if visible:
            center = torch.tensor(centers[0], dtype=torch.float32)
        else:
            center = torch.tensor([float("nan"),float("nan")])

        return {
            "frames": x,
            "heatmap": heatmap,
            "visible": torch.tensor(visible),
            "center": center,
            "clip": clip,
            "frame": t
        }


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    ds = BallDataset("train")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True, drop_last=True)
    print("batch per epoch:", len(loader))                   

    start = time.perf_counter()
    batch = next(iter(loader))                                  
    print("first batch in", round(time.perf_counter() - start, 1), "s")
    for k, v in batch.items():
        print(f"  {k:8s}", tuple(v.shape) if torch.is_tensor(v) else v)

    start = time.perf_counter()
    for n, batch in enumerate(loader):                         
        if n == 50:
            break
    print(f"{50 / (time.perf_counter() - start):.1f} batch/s")