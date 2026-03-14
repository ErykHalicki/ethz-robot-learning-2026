"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

from torch.utils.data import DataLoader, random_split

EPOCHS = 100
BATCH_SIZE = 256
LR = 2e-3
VAL_SPLIT = 0.2


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches+=1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        with torch.no_grad():
            loss = model.compute_loss(states, action_chunks)

        total_loss += loss.item()
        n_batches+=1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=200,
        help="Size of model hidden layers",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=24,
        help="number of model hidden layers",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS}).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.75)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path(f"./checkpoints/multi_cube/{ts}")
    else:
        ckpt_dir = Path(f"./checkpoints/single_cube/{ts}")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)


    def save_model(save_path):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalizer": {
                    "state_mean": normalizer.state_mean,
                    "state_std": normalizer.state_std,
                    #"action_mean": normalizer.action_mean,
                    #"action_std": normalizer.action_std,
                    #"state_mean": 0,
                    #"state_std": 1,
                    "action_mean": 0,
                    "action_std": 1,
                    # changed this because i want to
                    # discretize my action space, and it is getting in the way
                },
                "chunk_size": args.chunk_size,
                "policy_type": args.policy,
                "state_keys": args.state_keys,
                "action_keys": args.action_keys,
                "state_dim": int(states.shape[1]),
                "action_dim": int(actions.shape[1]),
                "val_loss": val_loss,
                "d_model": args.d_model,
                "depth": args.depth,
            },
            save_path,
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            save_name = f"best_model_{action_space}_{args.policy}.pt"
            save_path = ckpt_dir / save_name
            save_model(save_path)
            save_model(ckpt_dir.parent / "latest.pt")
            tag = " ✓ best"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
