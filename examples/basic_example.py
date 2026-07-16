import argparse
import time
from pathlib import Path
import sys

import timm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from slorr import (
    AdamSLORRHoyerDecoupled,
    slorr_hoyer_loss,
    slorr_nuc_loss,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def get_reg_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if name not in ["conv1", "fc", "patch_embed.proj", "head"]:
                layers.append(name)
    return layers


def get_wd_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def parse_args():
    parser = argparse.ArgumentParser("SLORR ImageNet example")
    parser.add_argument("--data", type=str, required=True, help="ImageNet root with train/ and val/")
    parser.add_argument(
        "--method",
        choices=["slorr_hoyer", "slorr_nuc", "slorr_hoyer_decoupled"],
        default="slorr_hoyer",
    )
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--slorr-lambda", type=float, default=1e-1)
    parser.add_argument("--pe-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="slorr_imagenet_example.pt")
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed + worker_id)


def evaluate(model, loader, device):
    model.eval()
    total, loss_sum = 0, 0.0
    correct1, correct5 = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

            pred1 = logits.argmax(dim=1)
            correct1 += (pred1 == y).sum().item()

            top5 = logits.topk(k=min(5, logits.size(1)), dim=1).indices
            correct5 += top5.eq(y.view(-1, 1)).any(dim=1).sum().item()

    return loss_sum / total, correct1 / total, correct5 / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    device = "cuda"

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(Path(args.data) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(args.data) / "val", transform=val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=1000).to(device)
    if args.pretrained_path:
        state = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state)

    reg_layers = get_reg_layers(model)
    print("Regularizing layers:", reg_layers)

    param_groups = get_wd_param_groups(model, weight_decay=args.wd)
    if args.method == "slorr_hoyer_decoupled":
        optimizer = AdamSLORRHoyerDecoupled(
            param_groups,
            model=model,
            layer_names=reg_layers,
            lr=args.lr,
            weight_decay=args.wd,
            slorr_lambda=args.slorr_lambda,
            steps=args.pe_steps,
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    main_steps = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=args.lr * 0.1)
    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = cosine

    val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
    print(
        f"epoch 0 val_loss {val_loss:.4f} val_top1 {val_top1:.4f} val_top5 {val_top5:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, total = 0.0, 0
        regloss_last = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            if args.method == "slorr_hoyer":
                regloss = slorr_hoyer_loss(
                    model,
                    layer_names=reg_layers,
                    reg_lambda=args.slorr_lambda,
                    steps=args.pe_steps,
                )
            elif args.method == "slorr_nuc":
                regloss = slorr_nuc_loss(
                    model,
                    layer_names=reg_layers,
                    reg_lambda=args.slorr_lambda,
                    steps=args.pe_steps,
                )
            elif args.method == "slorr_hoyer_decoupled":
                regloss = torch.tensor(0.0).to(device)
            else:
                raise ValueError(f"Unknown method {args.method}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.method == "slorr_hoyer_decoupled":
                regloss_last = float(optimizer.last_reg_loss)
            else:
                regloss_last = float(regloss)
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

        train_loss = loss_sum / total
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch} train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f} val_top1 {val_top1:.4f} val_top5 {val_top5:.4f} "
            f"time {elapsed:.1f}s regloss {regloss_last:.4f}"
        )

    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
