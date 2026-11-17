"""Turn raw SoccerNet-Tracking 2023 clips into a ball-detection dataset.

Outputs (under data/processed/ by default):
  labels.parquet  one row per (clip, frame, ball); frames without a ball get one row with visible=False
  clips.csv       one row per clip: split, game, action class, ball tracklets
  meta.json       resolutions, scale factors, split games, summary counts
  frames/<clip>/000001.jpg ...  frames resized to --width x --height
  preview/*.jpg   a few resized frames with the labels drawn on them, for a visual check

Usage (from the project root):
  .venv\\Scripts\\python scripts\\prepare_dataset.py
  .venv\\Scripts\\python scripts\\prepare_dataset.py --skip-frames      # labels only
"""
import argparse
import configparser
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=PROJECT / "data/raw/SoccerNet/tracking-2023")
    p.add_argument("--out", type=Path, default=PROJECT / "data/processed")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=352, help="keep it divisible by 16 for a 4-level U-Net")
    p.add_argument("--val-games", type=int, nargs="+", default=[7], help="gameIDs used for validation")
    p.add_argument("--test-games", type=int, nargs="+", default=[8, 11],
                   help="gameIDs used for test; every other game goes to train")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--previews", type=int, default=8)
    p.add_argument("--skip-frames", action="store_true", help="only rebuild the label files")
    p.add_argument("--overwrite", action="store_true", help="re-encode frames that already exist")
    return p.parse_args()


def read_ini(path: Path) -> configparser.SectionProxy:
    cfg = configparser.ConfigParser()  # note: keys come back lower-case ("trackletid_18")
    if not cfg.read(path):
        raise FileNotFoundError(path)
    return cfg["Sequence"]


def parse_clip(clip_dir: Path, raw_split: str, val_games: set[int], test_games: set[int]):
    """Return (frame-level ball rows, clip metadata, number of dropped out-of-image boxes)."""
    seq = read_ini(clip_dir / "seqinfo.ini")
    info = read_ini(clip_dir / "gameinfo.ini")
    W, H, L = int(seq["imWidth"]), int(seq["imHeight"]), int(seq["seqLength"])
    game_id = int(info["gameID"])

    # tracklet id -> label, keeping every ball tracklet (ball;1, ball;2, ball;none ...)
    balls = {}
    for key, value in info.items():
        value = value.strip()
        if key.startswith("trackletid_") and value.split(";")[0] == "ball":
            balls[int(key.split("_")[1])] = value

    gt = pd.read_csv(clip_dir / "gt" / "gt.txt", header=None, usecols=range(6),
                     names=["frame", "ball_id", "x", "y", "w", "h"])
    b = gt[gt["ball_id"].isin(list(balls)) & gt["frame"].between(1, L)]
    b = b.drop_duplicates(["frame", "ball_id"]).copy()
    b["ball_label"] = b["ball_id"].map(balls)
    b["cx"] = b["x"] + b["w"] / 2
    b["cy"] = b["y"] + b["h"] / 2
    b["size"] = (b["w"] + b["h"]) / 2
    inside = b["cx"].between(0, W, inclusive="left") & b["cy"].between(0, H, inclusive="left")
    dropped = int((~inside).sum())
    b = b[inside]

    # one row per frame even when there is no ball, so negatives are explicit
    frames = pd.DataFrame({"frame": np.arange(1, L + 1)})
    df = frames.merge(b, on="frame", how="left")
    df["visible"] = df["ball_id"].notna()
    df["n_balls"] = df.groupby("frame")["visible"].transform("sum").astype("int8")
    df["ball_id"] = df["ball_id"].astype("Int16")

    split = "test" if game_id in test_games else ("val" if game_id in val_games else "train")
    meta = dict(split=split, raw_split=raw_split, game_id=game_id, clip=clip_dir.name,
                action_class=info.get("actionClass", ""), game_time_start=info.get("gameTimeStart", ""),
                width=W, height=H, n_frames=L, fps=int(seq["frameRate"]),
                ball_tracklets=" | ".join(f"{k}:{v}" for k, v in sorted(balls.items())))
    for col in ("split", "game_id", "clip", "action_class"):
        df[col] = meta[col]
    return df, meta, dropped


def resize_one(job):
    src, dst, size, quality, overwrite = job
    if dst.exists() and not overwrite:
        return 0
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"cannot read {src}")
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # INTER_AREA: best for downscaling
    tmp = dst.with_name(dst.stem + ".tmp.jpg")                    # write then rename: safe to interrupt
    if not cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"cannot write {tmp}")
    os.replace(tmp, dst)
    return 1


def draw_previews(labels: pd.DataFrame, out: Path, n: int, seed: int = 0):
    """Draw the scaled labels on resized frames: if the circles sit on the ball, the whole chain is right."""
    prev = out / "preview"
    prev.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    with_ball = labels[labels["visible"]][["clip", "frame", "n_balls"]].drop_duplicates()
    picks = with_ball.sample(min(n - 1, len(with_ball)), random_state=seed)
    multi = with_ball[with_ball["n_balls"] > 1]
    if len(multi):
        picks = pd.concat([picks, multi.iloc[[rng.integers(len(multi))]]])
    for clip, frame in picks[["clip", "frame"]].itertuples(index=False):
        img = cv2.imread(str(out / "frames" / clip / f"{frame:06d}.jpg"))
        if img is None:
            continue
        rows = labels[(labels["clip"] == clip) & (labels["frame"] == frame) & labels["visible"]]
        for r in rows.itertuples():
            c = (int(round(r.cx_s)), int(round(r.cy_s)))
            cv2.circle(img, c, 10, (0, 0, 255), 2)
            cv2.putText(img, r.ball_label, (c[0] + 12, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(prev / f"{clip}_{frame:06d}.jpg"), img)


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clip_dirs = sorted((d for d in a.raw.glob("*/SNMOT-*") if d.is_dir()), key=lambda d: d.name)
    if not clip_dirs:
        raise SystemExit(f"no clips found under {a.raw}")

    # 1) labels -----------------------------------------------------------------
    parts, metas, dropped = [], [], 0
    for d in tqdm(clip_dirs, desc="labels"):
        df, meta, n_drop = parse_clip(d, d.parent.name, set(a.val_games), set(a.test_games))
        parts.append(df)
        metas.append(meta)
        dropped += n_drop
    labels = pd.concat(parts, ignore_index=True)
    clips = pd.DataFrame(metas)

    games = set(clips["game_id"])
    if set(a.val_games) & set(a.test_games):
        raise SystemExit("a game cannot be both in val and test")
    unknown = (set(a.val_games) | set(a.test_games)) - games
    if unknown:
        raise SystemExit(f"unknown gameIDs {sorted(unknown)}; available: {sorted(games)}")

    W, H = int(clips["width"].iloc[0]), int(clips["height"].iloc[0])
    if not ((clips["width"] == W) & (clips["height"] == H)).all():
        raise SystemExit("clips have different resolutions: per-clip scale factors would be needed")
    sx, sy = W / a.width, H / a.height
    labels["cx_s"] = labels["cx"] / sx
    labels["cy_s"] = labels["cy"] / sy
    labels["size_s"] = labels["size"] / ((sx + sy) / 2)

    cols = ["split", "game_id", "clip", "action_class", "frame", "n_balls", "visible",
            "ball_id", "ball_label", "x", "y", "w", "h", "cx", "cy", "size", "cx_s", "cy_s", "size_s"]
    labels = labels[cols].sort_values(["clip", "frame", "ball_id"], ignore_index=True)
    labels.to_parquet(a.out / "labels.parquet", index=False)
    clips.to_csv(a.out / "clips.csv", index=False)

    # 2) frames -----------------------------------------------------------------
    written = 0
    if not a.skip_frames:
        jobs = []
        for d in clip_dirs:
            dst_dir = a.out / "frames" / d.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for t in range(1, int(clips.loc[clips["clip"] == d.name, "n_frames"].iloc[0]) + 1):
                name = f"{t:06d}.jpg"
                jobs.append((d / "img1" / name, dst_dir / name, (a.width, a.height), a.jpeg_quality, a.overwrite))
        with ThreadPoolExecutor(a.workers) as ex:  # cv2 releases the GIL, so threads run in parallel
            for n in tqdm(ex.map(resize_one, jobs, chunksize=64), total=len(jobs), desc="frames"):
                written += n
        missing = [j[1] for j in jobs if not j[1].exists()]
        if missing:
            raise SystemExit(f"{len(missing)} resized frames missing, e.g. {missing[0]}")
        draw_previews(labels, a.out, a.previews)

    # 3) summary ----------------------------------------------------------------
    per_frame = labels.drop_duplicates(["clip", "frame"])
    summary = {}
    for split, g in per_frame.groupby("split"):
        balls = labels[(labels["split"] == split) & labels["visible"]]
        summary[split] = dict(
            clips=int(g["clip"].nunique()), games=sorted(int(x) for x in g["game_id"].unique()),
            frames=len(g), frames_with_ball=round(float((g["n_balls"] > 0).mean()), 4),
            frames_multi_ball=int((g["n_balls"] > 1).sum()),
            ball_size_median_px=float(balls["size"].median()),
            ball_size_median_px_resized=round(float(balls["size_s"].median()), 2))
    meta = dict(source_size=[W, H], size=[a.width, a.height], scale=[sx, sy],
                val_games=a.val_games, test_games=a.test_games,
                jpeg_quality=a.jpeg_quality, dropped_out_of_image_boxes=dropped,
                frame_path="frames/{clip}/{frame:06d}.jpg", splits=summary)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nlabels: {len(labels)} rows -> {a.out / 'labels.parquet'}")
    print(f"frames: {written} newly written" + (" (skipped)" if a.skip_frames else ""))
    print(f"dropped ball boxes with centre outside the image: {dropped}")
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
