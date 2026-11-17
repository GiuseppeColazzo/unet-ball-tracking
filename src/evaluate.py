import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PROJECT, BallDataset
from src.model import UNet
from src.train import BATCH_SIZE, DEVICE, NUM_WORKERS, TOLERANCE, count_results, find_ball


RUN_DIR = PROJECT / "runs" / "unet_focal_b_aug"
CHECKPOINT = RUN_DIR / "best.pt"
THRESHOLDS = np.arange(0.05, 0.65, 0.05)
SCALE = 3.0     # 640x352 -> 1920x1080 is about x3
TEST = "--test" in sys.argv



@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_xy, all_score, all_center, all_visible = [], [], [], []

    for batch in tqdm(loader, leave=False, dynamic_ncols=True, mininterval=1):
        x = batch["frames"].to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)

        pred_xy, score = find_ball(logits)
        all_xy.append(pred_xy.cpu())
        all_score.append(score.cpu())
        all_center.append(batch["center"])
        all_visible.append(batch["visible"])

    # run the model only once, then try every threshold on the saved predictions
    return torch.cat(all_xy), torch.cat(all_score), torch.cat(all_center), torch.cat(all_visible)


def metrics(pred_xy, score, center, visible, threshold):
    tp, fp, fn = count_results(pred_xy, score, center, visible, threshold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, tp, fp, fn



def main():
    split = "test" if TEST else "val"
    dataset = BallDataset(split)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print(f"checkpoint {CHECKPOINT.relative_to(PROJECT)} | {split}: {len(dataset)} samples")

    pred_xy, score, center, visible = predict(model, loader)

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

    # test: use the threshold chosen on val, never tune it here
    threshold = json.loads((RUN_DIR / "threshold.json").read_text())["threshold"]
    p, r, f1, tp, fp, fn = metrics(pred_xy, score, center, visible, threshold)

    dist = (pred_xy - center).norm(dim=1)
    correct = visible & (score >= threshold) & (dist <= TOLERANCE)
    mean_dist = dist[correct].mean().item()

    print(f"\nTEST (threshold {threshold})")
    print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  | frames with ball {int(visible.sum())} / {len(visible)}")
    print(f"  mean error on correct detections: {mean_dist:.2f} px (~{mean_dist * SCALE:.1f} px in 1920x1080)")


if __name__ == "__main__":
    main()
