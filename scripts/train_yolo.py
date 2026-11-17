"""Single-frame YOLO baseline, trained on the same games as the U-Net.

    .venv\\Scripts\\python -m scripts.train_yolo

Uses the raw 1920x1080 frames resized by YOLO to 1280 (the ball is ~10 px there), the usual
off-the-shelf recipe for small objects. Results go in runs/yolo11s_1280/.
"""
from ultralytics import YOLO

from src.dataset import PROJECT

WEIGHTS = PROJECT / "weights" / "yolo11s.pt"    # COCO-pretrained, downloaded by Ultralytics on first use


def main():
    WEIGHTS.parent.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(PROJECT / "data" / "yolo" / "data.yaml"),
        imgsz=1280,
        epochs=10,
        batch=4,            # 8 GB of GPU memory
        workers=4,
        single_cls=True,
        close_mosaic=2,     # default (10) would switch mosaic augmentation off for the whole run
        project=str(PROJECT / "runs"),
        name="yolo11s_1280",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
