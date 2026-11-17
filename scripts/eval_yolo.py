"""Evaluate the YOLO baseline with the same frames and the same metric as the U-Net.

    .venv\\Scripts\\python -m scripts.eval_yolo           # threshold sweep on val
    .venv\\Scripts\\python -m scripts.eval_yolo --test    # test with the threshold chosen on val

For every frame the most confident box is the prediction; its centre is converted to the
640x352 coordinates used by src.evaluate, so tolerance and counting are identical.
"""
import json
import sys

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.dataset import PROCESSED, PROJECT, BallDataset
from src.train import TOLERANCE, count_results

RUN_DIR = PROJECT / "runs" / "yolo11s_1280"
RAW = PROJECT / "data" / "raw" / "SoccerNet" / "tracking-2023"
THRESHOLDS = np.arange(0.05, 0.90, 0.05)
BATCH = 16
TEST = "--test" in sys.argv


def predict(model, dataset):
    sx, sy = json.loads((PROCESSED / "meta.json").read_text())["scale"]
    clip_dirs = {d.name: d for d in RAW.glob("*/SNMOT-*")}
    paths = [str(clip_dirs[clip] / "img1" / f"{t:06d}.jpg") for clip, t in dataset.samples]

    xy, score = [], []
    for i in tqdm(range(0, len(paths), BATCH), dynamic_ncols=True, mininterval=1):
        results = model.predict(paths[i:i + BATCH], imgsz=1280, conf=0.01, max_det=5, half=True, verbose=False)
        for r in results:
            if len(r.boxes) == 0:
                xy.append([0.0, 0.0])
                score.append(0.0)
                continue
            best = r.boxes.conf.argmax()
            cx, cy = r.boxes.xywh[best, :2].tolist()        # pixels in the 1920x1080 frame
            xy.append([cx / sx, cy / sy])
            score.append(r.boxes.conf[best].item())
    return torch.tensor(xy), torch.tensor(score)


def metrics(pred_xy, score, center, visible, threshold):
    tp, fp, fn = count_results(pred_xy, score, center, visible, threshold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, tp, fp, fn


def main():
    split = "test" if TEST else "val"
    dataset = BallDataset(split)
    model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
    print(f"{RUN_DIR.name} | {split}: {len(dataset)} samples")

    # same ground truth as the U-Net evaluation: first annotated ball of frame t
    empty = np.zeros((0, 2), dtype=np.float32)
    centers = [dataset.centers.get(s, empty) for s in dataset.samples]
    visible = torch.tensor([len(c) > 0 for c in centers])
    center = torch.tensor(np.array([c[0] if len(c) else [np.nan, np.nan] for c in centers]), dtype=torch.float32)

    pred_xy, score = predict(model, dataset)

    if not TEST:
        print("\nthreshold  precision  recall  F1")
        best_f1, best_threshold = 0, None
        for threshold in THRESHOLDS:
            p, r, f1, _, _, _ = metrics(pred_xy, score, center, visible, threshold)
            print(f"  {threshold:.2f}       {p:.3f}    {r:.3f}  {f1:.3f}")
            if f1 > best_f1:
                best_f1, best_threshold = f1, round(float(threshold), 2)
        (RUN_DIR / "threshold.json").write_text(json.dumps({"threshold": best_threshold, "val_f1": best_f1}))
        print(f"\nbest threshold {best_threshold} (val F1 {best_f1:.3f}) -> saved in threshold.json")
        return

    threshold = json.loads((RUN_DIR / "threshold.json").read_text())["threshold"]
    p, r, f1, tp, fp, fn = metrics(pred_xy, score, center, visible, threshold)
    dist = (pred_xy - center).norm(dim=1)
    correct = visible & (score >= threshold) & (dist <= TOLERANCE)
    print(f"\nTEST (threshold {threshold})")
    print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  | frames with ball {int(visible.sum())} / {len(visible)}")
    print(f"  mean error on correct detections: {dist[correct].mean().item():.2f} px (640x352)")


if __name__ == "__main__":
    main()
