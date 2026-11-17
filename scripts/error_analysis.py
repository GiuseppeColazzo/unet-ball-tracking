import configparser
import json

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PROCESSED, PROJECT, BallDataset
from src.model import UNet
from src.train import BATCH_SIZE, DEVICE, NUM_WORKERS, TOLERANCE

RUN_DIR = PROJECT / "runs" / "unet_focal_b_aug"
RAW = PROJECT / "data" / "raw" / "SoccerNet" / "tracking-2023"
SPLIT = "val"
TOPK = 3


@torch.no_grad()
def predict(model, loader):
    """Top-3 peaks with the real 3 frames, and the top peak when frame t is repeated 3 times (no motion)."""
    model.eval()
    rows = []
    for batch in tqdm(loader, dynamic_ncols=True, mininterval=1):
        x = batch["frames"].to(DEVICE, non_blocking=True)
        still = x[:, 3:6].repeat(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            prob = torch.sigmoid(model(x)[:, 0].float())
            prob_still = torch.sigmoid(model(still)[:, 0].float())

        W = prob.shape[-1]
        peaks = prob * (prob == F.max_pool2d(prob[:, None], 9, 1, 4)[:, 0])   # keep local maxima only
        score, idx = peaks.flatten(1).topk(TOPK, dim=1)
        xs, ys = (idx % W).float() + 0.5, (idx // W).float() + 0.5
        s_score, s_idx = prob_still.flatten(1).max(dim=1)

        for b in range(len(x)):
            row = {"clip": batch["clip"][b], "frame": int(batch["frame"][b])}
            for k in range(TOPK):
                row[f"x{k + 1}"], row[f"y{k + 1}"], row[f"s{k + 1}"] = xs[b, k].item(), ys[b, k].item(), score[b, k].item()
            row["still_x"] = (s_idx[b] % W).item() + 0.5
            row["still_y"] = (s_idx[b] // W).item() + 0.5
            row["still_s"] = s_score[b].item()
            rows.append(row)
    return pd.DataFrame(rows)


def person_overlap(labels):
    """For every ball row: is the ball centre inside the box of a player / goalkeeper / referee?"""
    out = []
    for clip, g in tqdm(labels[labels["visible"]].groupby("clip"), desc="person boxes"):
        clip_dir = next(RAW.glob(f"*/{clip}"))
        cfg = configparser.ConfigParser()
        cfg.read(clip_dir / "gameinfo.ini")
        people = [int(k.split("_")[1]) for k, v in cfg["Sequence"].items()
                  if k.startswith("trackletid_") and v.strip().split(";")[0] != "ball"]
        gt = pd.read_csv(clip_dir / "gt" / "gt.txt", header=None, usecols=range(6),
                         names=["frame", "id", "x", "y", "w", "h"])
        gt = gt[gt["id"].isin(people)]
        boxes = {f: b[["x", "y", "w", "h"]].to_numpy() for f, b in gt.groupby("frame")}
        for r in g.itertuples():
            b = boxes.get(r.frame)
            inside = False
            if b is not None:
                inside = bool(((b[:, 0] <= r.cx) & (r.cx <= b[:, 0] + b[:, 2]) &
                               (b[:, 1] <= r.cy) & (r.cy <= b[:, 1] + b[:, 3])).any())
            out.append((clip, r.frame, r.ball_id, inside))
    return pd.DataFrame(out, columns=["clip", "frame", "ball_id", "on_person"])


def prf(df, thr):
    """Same metric as src.evaluate, using the first annotated ball."""
    found = df["s1"] >= thr
    correct = df["visible"] & found & (df["dist_first"] <= TOLERANCE)
    tp, fp, fn = correct.sum(), (found & ~correct).sum(), (df["visible"] & ~correct).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def table(df, by, thr):
    rows = []
    for key, g in df.groupby(by, observed=True):
        v = g[g["visible"]]
        rows.append({by: key, "frames": len(v), "recall": (v["outcome"] == "correct").mean(),
                     "wrong_place": (v["outcome"] == "wrong_place").mean(),
                     "low_conf": v["outcome"].isin(["low_conf_right", "low_conf_wrong"]).mean(),
                     "in_top3": v["in_top3"].mean()})
    return pd.DataFrame(rows).round(3).to_string(index=False)


def save_examples(df, name, n=12, crop=48, zoom=4):
    """Crops around the true ball (green) with the predicted peak (red) if it falls inside the crop."""
    out = RUN_DIR / "errors"
    out.mkdir(exist_ok=True)
    tiles = []
    for r in df.sample(min(n, len(df)), random_state=0).itertuples():
        img = cv2.imread(str(PROCESSED / "frames" / r.clip / f"{r.frame:06d}.jpg"))
        H, W = img.shape[:2]
        x0 = int(np.clip(r.cx_s - crop, 0, W - 2 * crop))
        y0 = int(np.clip(r.cy_s - crop, 0, H - 2 * crop))
        tile = cv2.resize(img[y0:y0 + 2 * crop, x0:x0 + 2 * crop], None, fx=zoom, fy=zoom,
                          interpolation=cv2.INTER_NEAREST)
        cv2.circle(tile, (int((r.cx_s - x0) * zoom), int((r.cy_s - y0) * zoom)), 22, (0, 255, 0), 1)
        px, py = (r.x1 - x0) * zoom, (r.y1 - y0) * zoom
        if 0 <= px < tile.shape[1] and 0 <= py < tile.shape[0]:
            cv2.circle(tile, (int(px), int(py)), 14, (0, 0, 255), 1)
        cv2.putText(tile, f"{r.clip[-3:]}/{r.frame} s={r.s1:.2f} d={r.dist:.0f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out / f"{name}.jpg"), grid)


def main():
    thr = json.loads((RUN_DIR / "threshold.json").read_text())["threshold"]
    pred_path = RUN_DIR / f"predictions_{SPLIT}.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
    else:
        ds = BallDataset(SPLIT)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        model = UNet().to(DEVICE)
        model.load_state_dict(torch.load(RUN_DIR / "best.pt", map_location=DEVICE))
        pred = predict(model, loader)
        pred.to_parquet(pred_path, index=False)

    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    labels = labels[labels["split"] == SPLIT]
    labels = labels.merge(person_overlap(labels), on=["clip", "frame", "ball_id"], how="left")

    # ball speed in pixels (640x352) per frame, from the first ball of the neighbouring frames
    first = labels.drop_duplicates(["clip", "frame"]).sort_values(["clip", "frame"]).copy()
    g = first.groupby("clip")
    first["speed"] = np.hypot(g["cx_s"].shift(-1) - g["cx_s"].shift(1), g["cy_s"].shift(-1) - g["cy_s"].shift(1)) / 2

    # distance from each of the top-3 peaks to the nearest annotated ball
    balls = labels[labels["visible"]][["clip", "frame", "cx_s", "cy_s"]]
    m = pred.merge(balls, on=["clip", "frame"], how="left")
    for k in range(1, TOPK + 1):
        m[f"d{k}"] = np.hypot(m[f"x{k}"] - m["cx_s"], m[f"y{k}"] - m["cy_s"])
    m["d_still"] = np.hypot(m["still_x"] - m["cx_s"], m["still_y"] - m["cy_s"])
    near = m.groupby(["clip", "frame"])[["d1", "d2", "d3", "d_still"]].min().reset_index()

    df = pred.merge(first, on=["clip", "frame"]).merge(near, on=["clip", "frame"])
    df["dist_first"] = np.hypot(df["x1"] - df["cx_s"], df["y1"] - df["cy_s"])
    df["dist"] = df["d1"]
    found, close = df["s1"] >= thr, df["d1"] <= TOLERANCE
    df["outcome"] = np.select(
        [~df["visible"] & found, ~df["visible"], found & close, found & ~close, ~found & close],
        ["false_alarm", "true_negative", "correct", "wrong_place", "low_conf_right"], "low_conf_wrong")
    df["in_top3"] = ((df["d1"] <= TOLERANCE) & (df["s1"] >= 0.05)) | ((df["d2"] <= TOLERANCE) & (df["s2"] >= 0.05)) | \
                    ((df["d3"] <= TOLERANCE) & (df["s3"] >= 0.05))
    vis = df[df["visible"]]

    print(f"\n=== {SPLIT}: {len(df)} frames, {len(vis)} with ball, threshold {thr}, tolerance {TOLERANCE} px ===")
    p, r, f1 = prf(df, thr)
    print(f"official metric (first ball): P {p:.3f} R {r:.3f} F1 {f1:.3f}")

    print("\n--- outcomes (frames with ball) ---")
    print((vis["outcome"].value_counts(normalize=True)).round(3).to_string())
    print("\n--- outcomes (frames without ball) ---")
    print(df[~df["visible"]]["outcome"].value_counts().to_string())

    print("\n--- ball among the top-3 peaks (score >= 0.05): upper bound for a tracker ---")
    print(f"top-1 within {TOLERANCE}px: {(vis['d1'] <= TOLERANCE).mean():.3f} | top-3: {vis['in_top3'].mean():.3f}")

    print("\n--- wrong_place: how far is the peak from the ball (px 640x352) ---")
    wp = vis[vis["outcome"] == "wrong_place"]
    print(pd.cut(wp["dist"], [4, 8, 20, 100, 1000]).value_counts(normalize=True, sort=False).round(3).to_string())

    print("\n--- no motion: frame t repeated 3 times (same model, no retraining) ---")
    still = df.assign(s1=df["still_s"], dist_first=np.hypot(df["still_x"] - df["cx_s"], df["still_y"] - df["cy_s"]))
    p, r, f1 = prf(still, thr)
    print(f"P {p:.3f} R {r:.3f} F1 {f1:.3f}")

    df["size_bin"] = pd.cut(df["size_s"], [0, 3.5, 5, 7, 100], labels=["<3.5px", "3.5-5px", "5-7px", ">7px"])
    df["speed_bin"] = pd.cut(df["speed"], [-1, 0.3, 2, 6, 1000], labels=["still <0.3", "slow 0.3-2", "medium 2-6", "fast >6"])
    df["balls"] = df["n_balls"].clip(upper=2)
    for by in ["size_bin", "speed_bin", "on_person", "balls", "action_class"]:
        print(f"\n--- by {by} ---")
        print(table(df, by, thr))

    print("\n--- error streaks: consecutive frames with ball that are not 'correct' ---")
    runs = []
    for _, c in df.sort_values("frame").groupby("clip"):
        bad = (c["visible"] & (c["outcome"] != "correct")).to_numpy()
        n = 0
        for b in list(bad) + [False]:
            if b:
                n += 1
            elif n:
                runs.append(n)
                n = 0
    runs = pd.Series(runs)
    bins = pd.cut(runs, [0, 2, 5, 12, 25, 10000], labels=["1-2", "3-5", "6-12", "13-25 (0.5-1s)", ">25 (>1s)"])
    print((runs.groupby(bins, observed=True).sum() / runs.sum()).round(3).rename("share of error frames").to_string())

    print("\n--- per clip (worst 6 and best 3 by recall) ---")
    per_clip = vis.groupby("clip").agg(frames=("frame", "size"), recall=("outcome", lambda s: (s == "correct").mean()),
                                       action=("action_class", "first"), size=("size_s", "median")).round(3)
    print(pd.concat([per_clip.sort_values("recall").head(6), per_clip.sort_values("recall").tail(3)]).to_string())

    for name in ["wrong_place", "low_conf_wrong", "low_conf_right", "false_alarm"]:
        sub = df[df["outcome"] == name]
        if name == "false_alarm":
            sub = sub.assign(cx_s=sub["x1"], cy_s=sub["y1"], dist=0.0)
        if len(sub):
            save_examples(sub, name)
    print(f"\nexample images in {RUN_DIR / 'errors'}")


if __name__ == "__main__":
    main()
