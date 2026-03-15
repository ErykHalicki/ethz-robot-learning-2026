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
from itertools import permutations as _permutations
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

_CUBE_PERMS = [list(p) for p in _permutations([0, 1, 2])]


def augment_multicube_permutations(
    states: torch.Tensor,
    action_chunks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all 6 cube-slot permutations of each sample.

    State layout (last 12 dims before the goal one-hot):
        [..., -12:-9]: red cube xyz
        [...,  -9:-6]: green cube xyz
        [...,  -6:-3]: blue cube xyz
        [...,   -3:  ]: goal one-hot (3,)

    Output shapes: (6*B, D) and (6*B, H, A).
    """
    # generate fake data by swapping around the goal cube state vectors based on goal
    # NOTE: REQUIRES SPECIFIC STATE VECTOR ORDER
    # For every sample we can create 6 artificial samples:
    #   Base sample (no change)
    #   swap non-goal positions, keep goal the same ([1,2,3] -> [3,2,1] for goal [0, 1, 0])
    #   swap goal with non-goal (positions and goal) ([1,2,3] -> [2,1,3] for goal [0, 1, 0] -> [1, 0, 0])
    #       swap goal with non-goal AND non-goal positions (123 -> 231) for goal [010] -> [100]
    aug_states = []
    for perm in _CUBE_PERMS:
        s = states.clone()
        cube_pos = s[..., -12:-3].reshape(*s.shape[:-1], 3, 3)
        s[..., -12:-3] = cube_pos[..., perm, :].reshape(*s.shape[:-1], 9)
        s[..., -3:] = states[..., -3:][..., perm]
        aug_states.append(s)
    return torch.cat(aug_states, dim=0), torch.cat([action_chunks] * 6, dim=0)


EPOCHS = 100
BATCH_SIZE = 512
LR = 4e-3
VAL_SPLIT = 0.2


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    multicube=False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        states, action_chunks = batch
        if multicube:
            
            states, action_chunks = augment_multicube_permutations(states, action_chunks)

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

    parser.add_argument(
        "--multicube",
        action="store_true",
        help="Generate fake data for multicube (REQUIRES SPECIFIC STATE VECTOR ORDER)"
        '"original_pos_cube_red[:3]"  "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal'
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
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)

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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.multicube)
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
