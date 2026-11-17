"""Build a YOLO-format dataset for the single-frame baseline.

    .venv\\Scripts\\python -m scripts.prepare_yolo

data/yolo/images/{train,val}/<clip>_<frame>.jpg   hard links to the raw 1920x1080 frames (no extra disk space)
data/yolo/labels/{train,val}/<clip>_<frame>.txt   "0 cx cy w h" normalised, one line per ball (empty = no ball)
data/yolo/data.yaml

Consecutive frames are almost identical, so train uses every 3rd frame and val every 5th
(YOLO's own validation during training). The final comparison is done by scripts/eval_yolo.py
on exactly the same frames and with the same metric as the U-Net.
"""
import os
import shutil

import pandas as pd
from tqdm import tqdm

from src.dataset import PROCESSED, PROJECT

RAW = PROJECT / "data" / "raw" / "SoccerNet" / "tracking-2023"
OUT = PROJECT / "data" / "yolo"
STEP = {"train": 3, "val": 5}
W, H = 1920, 1080


def main():
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    clip_dirs = {d.name: d for d in RAW.glob("*/SNMOT-*")}

    for split, step in STEP.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        part = labels[(labels["split"] == split) & (labels["frame"] % step == 0)]

        n_img, n_ball = 0, 0
        for (clip, frame), g in tqdm(part.groupby(["clip", "frame"]), desc=split):
            name = f"{clip}_{frame:06d}"
            src = clip_dirs[clip] / "img1" / f"{frame:06d}.jpg"
            dst = OUT / "images" / split / f"{name}.jpg"
            if not dst.exists():
                try:
                    os.link(src, dst)           # hard link: same file, no copy
                except OSError:
                    shutil.copy2(src, dst)

            balls = g[g["visible"]]
            lines = [f"0 {r.cx / W:.6f} {r.cy / H:.6f} {r.w / W:.6f} {r.h / H:.6f}" for r in balls.itertuples()]
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            n_img += 1
            n_ball += len(lines)
        print(f"{split}: {n_img} images, {n_ball} ball boxes")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    print(f"written {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
