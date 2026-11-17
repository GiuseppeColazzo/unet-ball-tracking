"""Train the U-Net for ball detection.

    .venv\\Scripts\\python -m src.train --overfit   # sanity check on a single batch
    .venv\\Scripts\\python -m src.train             # full training
"""
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import PROJECT, BallDataset
from src.model import UNet


LEARNING_RATE = 1e-3
BATCH_SIZE = 8
NUM_EPOCHS = 6
NUM_WORKERS = 4
THRESHOLD = 0.5    
TOLERANCE = 4.0  
DEVICE = "cuda"
RUN_DIR = PROJECT / "runs" / "unet_focal_b_aug"     # split B (train games 4, 6, 9) + augmentation
OVERFIT = "--overfit" in sys.argv



def focal_loss(logits, target, alpha=2, beta=4):
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.9     # pixels on the ball peak
    neg = ~pos

    pos_loss = -((1 - p) ** alpha * torch.log(p))[pos].sum()
    # (1 - target)^beta: small penalty for pixels close to the ball
    neg_loss = -((1 - target) ** beta * p ** alpha * torch.log(1 - p))[neg].sum()

    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


def find_ball(logits):
    prob = torch.sigmoid(logits[:, 0].float())      # (B, H, W)
    width = prob.shape[-1]

    score, idx = prob.flatten(1).max(dim=1)         # peak value and position
    x = (idx % width).float() + 0.5
    y = (idx // width).float() + 0.5

    return torch.stack([x, y], dim=1), score        # (B, 2), (B,)


def count_results(pred_xy, score, center, visible, threshold=THRESHOLD):
    found = score >= threshold
    dist = (pred_xy - center).norm(dim=1)           # nan when there is no ball
    correct = visible & found & (dist <= TOLERANCE)

    tp = correct.sum().item()
    fp = (found & ~correct).sum().item()            # found something that is not the ball
    fn = (visible & ~correct).sum().item()          # missed the ball

    return tp, fp, fn


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0

    loop = tqdm(loader, leave=False, dynamic_ncols=True, mininterval=1)    # fit the terminal width
    for batch in loop:
        x = batch["frames"].to(DEVICE, non_blocking=True)
        target = batch["heatmap"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = focal_loss(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0
    tp, fp, fn = 0, 0, 0

    for batch in tqdm(loader, leave=False, dynamic_ncols=True, mininterval=1):
        x = batch["frames"].to(DEVICE, non_blocking=True)
        target = batch["heatmap"].to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        total_loss += focal_loss(logits, target).item()

        pred_xy, score = find_ball(logits)
        t, f, n = count_results(pred_xy.cpu(), score.cpu(), batch["center"], batch["visible"])
        tp, fp, fn = tp + t, fp + f, fn + n

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return total_loss / len(loader), precision, recall, f1


def overfit_one_batch(model, dataset, optimizer, scaler, steps=300):
    batch = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)))
    x = batch["frames"].to(DEVICE)
    target = batch["heatmap"].to(DEVICE)

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = focal_loss(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 25 == 0:
            pred_xy, score = find_ball(logits)
            tp, fp, fn = count_results(pred_xy.cpu(), score.cpu(), batch["center"], batch["visible"])
            print(f"step {step}  loss {loss.item():.4f}  TP {tp}  FP {fp}  FN {fn}")



def main():
    train_ds = BallDataset("train", augment=True)

    model = UNet().to(DEVICE)
    model.final_conv.bias.data.fill_(-2.19)     # start with p ~ 0.1 everywhere (CenterNet trick)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")       # keeps float16 gradients from becoming zero

    if OVERFIT:
        overfit_one_batch(model, train_ds, optimizer, scaler)
        return

    val_ds = BallDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, persistent_workers=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(RUN_DIR)
    best_f1 = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_loss, precision, recall, f1 = validate(model, val_loader)
        scheduler.step()

        minutes = (time.perf_counter() - start) / 60
        print(f"epoch {epoch}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  ({minutes:.1f} min)")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("val/f1", f1, epoch)
        writer.add_scalar("val/precision", precision, epoch)
        writer.add_scalar("val/recall", recall, epoch)

        torch.save(model.state_dict(), RUN_DIR / "last.pt")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), RUN_DIR / "best.pt")

    writer.close()


if __name__ == "__main__":
    main()
