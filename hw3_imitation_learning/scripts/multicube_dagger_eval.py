"""DAgger interactive evaluation for the multicube scene.

Runs policy inference in the multicube goal-conditioned scene. At any time you
(the expert) can press the takeover key to assume manual control. While in
takeover mode every timestep is recorded into a zarr store (same format as
record_teleop_demos.py --multicube). Pressing the takeover key again or ending
the episode hands control back to the policy.

Collected data is saved under datasets/raw/multi_cube/dagger/ and can later be
merged with the original demonstrations for retraining via compute_actions.py.

Usage:
    python scripts/multicube_dagger_eval.py \\
        --checkpoint checkpoints/multi_cube/best_model.pt \\
        --num-episodes 10 \\
        --goal-cube all
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from hw3.dataset import Normalizer
from hw3.eval_utils import (
    apply_action,
    check_cube_out_of_bounds,
    check_success,
    check_wrong_cube_in_bin,
    infer_action_chunk,
    load_checkpoint,
)
from hw3.sim_env import (
    CUBE_COLORS,
    GOAL_DIM,
    SO100MulticubeSimEnv,
)
from hw3.teleop_utils import (
    CAMERA_NAMES,
    DEFAULT_KEYMAP_PATH,
    JOINT_NAMES,
    ZarrEpisodeWriter,
    compose_camera_views,
    handle_teleop_key,
    load_keymap,
)
from scripts.record_teleop_demos import MulticubeZarrWriter
from so101_gym.constants import ASSETS_DIR

XML_PATH_MULTICUBE = ASSETS_DIR / "so100_multicube_ee.xml"


def build_goal_schedule(goal_cube: str, num_episodes: int) -> list[str]:
    if goal_cube == "all":
        return [CUBE_COLORS[i % len(CUBE_COLORS)] for i in range(num_episodes)]
    return [goal_cube] * num_episodes


def run_dagger_episode(
    env: SO100MulticubeSimEnv,
    model: torch.nn.Module,
    normalizer: Normalizer,
    state_keys: list[str],
    action_keys: list[str],
    device: torch.device,
    writer: MulticubeZarrWriter,
    key_to_action: dict[int, str],
    *,
    max_steps: int = 800,
    successes: int = 0,
    total: int = 0,
    headless: bool = False,
) -> tuple[bool, int, bool, bool]:
    """Run one DAgger episode in the multicube scene.

    Returns (success, n_takeover_steps, aborted, replay).
    """
    rng_state_before_reset = env.rng.bit_generator.state
    obs = env.reset()

    action_queue: list[np.ndarray] = []
    step = 0
    success = False
    human_control = False
    n_takeover_steps = 0
    recording_this_episode = False
    GRACE_SECS = 1.7
    grace_steps_remaining: int | None = None

    while step < max_steps or human_control:
        if not headless:
            k_raw = cv2.waitKeyEx(1)
        else:
            k_raw = -1

        if k_raw != -1:
            action_name = key_to_action.get(k_raw)

            if action_name == "escape":
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded on escape.")
                return success, n_takeover_steps, True, False

            if action_name == "record":
                human_control = not human_control
                if human_control:
                    print("  >>> HUMAN TAKEOVER — you are now controlling the arm")
                    print("      Press your 'record' key again to hand back to policy")
                    action_queue.clear()
                    recording_this_episode = True
                else:
                    print("  <<< POLICY RESUMED")

            if action_name in ("goal_cube_red", "goal_cube_green", "goal_cube_blue"):
                if recording_this_episode:
                    print("  Cannot change goal cube while recording!")
                else:
                    goal_map = {"goal_cube_red": 0, "goal_cube_green": 1, "goal_cube_blue": 2}
                    idx = goal_map[action_name]
                    env.set_goal(CUBE_COLORS[idx])
                    print(f"  Goal cube set to: {CUBE_COLORS[idx]}")

            if action_name == "reset":
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — replaying same scenario.")
                env.rng.bit_generator.state = rng_state_before_reset
                return False, 0, False, True

            if k_raw == 13 or k_raw == 0x0D:
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — skipping to next.")
                return False, 0, False, False

            if human_control and action_name is not None:
                handle_teleop_key(
                    action_name,
                    env.data,
                    env.model,
                    env.mocap_id,
                    env.act_ids[env._jaw_idx],
                )

        if human_control:
            joints = env.get_joint_angles()
            ee_state = env.get_ee_state()
            all_cubes_state = env.get_all_cubes_state()
            gripper_state = np.array([env.get_gripper_angle()], dtype=np.float32)
            action_gripper = np.array(
                [env.data.ctrl[env.act_ids[env._jaw_idx]]], dtype=np.float32
            )
            dummy_obstacle = np.zeros(3, dtype=np.float32)
            goal_onehot = env.get_goal_onehot().astype(np.float32)
            goal_pos = env.get_goal_pos().astype(np.float32)
            writer.append_with_goal(
                joints,
                ee_state,
                all_cubes_state,
                gripper_state,
                action_gripper,
                dummy_obstacle,
                goal_onehot,
                goal_pos,
            )
            n_takeover_steps += 1

        if not human_control:
            if not action_queue:
                chunk = infer_action_chunk(
                    model=model,
                    normalizer=normalizer,
                    obs=obs,
                    state_keys=state_keys,
                    device=device,
                )
                action_queue.extend(chunk)

            action = action_queue.pop(0)
            apply_action(env, action, action_keys)

        obs = env.step()
        step += 1

        success = check_success(env)
        wrong_in_bin = check_wrong_cube_in_bin(env)

        if success and grace_steps_remaining is None:
            grace_steps_remaining = model.chunk_size
            print(
                f"  [{env.goal_cube}] Cube in bin! Recording {grace_steps_remaining} more "
                f"steps ({GRACE_SECS}s grace period)..."
            )

        if grace_steps_remaining is not None:
            grace_steps_remaining -= 1
            if grace_steps_remaining <= 0:
                if recording_this_episode:
                    writer.end_episode()
                    print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
                return True, n_takeover_steps, False, False

        if check_cube_out_of_bounds(env):
            print(f"  [{env.goal_cube}] Cube out of bounds — early termination.")
            if recording_this_episode:
                writer.discard_episode()
                print("  Episode discarded — replaying same scenario.")
                env.rng.bit_generator.state = rng_state_before_reset
                return False, 0, False, True

        if headless:
            continue

        img = compose_camera_views({cam: env.render(cam) for cam in CAMERA_NAMES})
        status = f"Goal: {env.goal_cube} | Step {step}/{max_steps}"
        if human_control:
            status += " | HUMAN CONTROL"
        else:
            status += f" | POLICY (queue {len(action_queue)})"
        cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if total > 0:
            rate = successes / total * 100
            sr_text = f"Success: {successes}/{total} ({rate:.0f}%)"
        else:
            sr_text = "Success: -/-"
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(img, sr_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        dagger_text = (
            f"DAgger steps: {n_takeover_steps} | Episodes saved: {writer.num_episodes}"
        )
        cv2.putText(img, dagger_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if wrong_in_bin:
            cv2.putText(
                img,
                f"WRONG CUBE IN BIN: {wrong_in_bin}!",
                (10, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        elif human_control:
            cv2.putText(img, "HUMAN", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(img, "POLICY", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        def _label_for(act):
            for code, a in key_to_action.items():
                if a == act:
                    if 32 <= (code & 0xFF) <= 126:
                        ch = chr(code & 0xFF)
                        return ch if ch.strip() else "SPACE"
                    if code & 0xFF == 27:
                        return "ESC"
                    return f"key:{code}"
            return "?"

        hint = (
            f"{_label_for('record')} takeover | "
            f"{_label_for('reset')} replay | "
            f"ENTER skip | "
            f"{_label_for('escape')} quit | "
            f"{_label_for('goal_cube_red')}/"
            f"{_label_for('goal_cube_green')}/"
            f"{_label_for('goal_cube_blue')} goal"
        )
        cv2.putText(
            img,
            hint,
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("DAgger Multicube Eval", img)
        if human_control:
            time.sleep(0.03)

    if recording_this_episode:
        writer.end_episode()
        print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
    return success, n_takeover_steps, False, False


def main():
    parser = argparse.ArgumentParser(
        description="DAgger interactive evaluation for multicube with human takeover."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible cube spawns.",
    )
    parser.add_argument(
        "--goal-cube",
        type=str,
        default="all",
        choices=["red", "green", "blue", "all"],
        help="Goal colour for multicube ('all' cycles evenly, default: all).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable multicube slot shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for DAgger zarr data (default: datasets/raw/multi_cube/dagger/<timestamp>).",
    )
    parser.add_argument(
        "--keymap",
        type=Path,
        default=None,
        help="Path to keymap.json (default: hw3/keymap.json).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering or real-time pacing. No human takeover in this mode.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, normalizer, chunk_size, state_keys, action_keys = load_checkpoint(
        args.checkpoint, device
    )

    use_mocap = not any("action_joints" in k for k in action_keys)

    goal_schedule = build_goal_schedule(args.goal_cube, args.num_episodes)

    print(f"Scene: {XML_PATH_MULTICUBE.name}")
    env = SO100MulticubeSimEnv(
        xml_path=XML_PATH_MULTICUBE,
        render_w=640,
        render_h=480,
        use_mocap=use_mocap,
        goal_cube=goal_schedule[0],
        shuffle_cubes=not args.no_shuffle,
        seed=args.seed,
        headless=args.headless,
    )

    km_path = args.keymap or DEFAULT_KEYMAP_PATH
    key_to_action = load_keymap(km_path)
    print(f"Loaded keymap from {km_path}")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("./datasets/raw/multi_cube/dagger") / ts
    out_zarr = out_dir / "so100_multicube_teleop.zarr"
    print(f"DAgger data will be saved to: {out_zarr}")

    writer = MulticubeZarrWriter(
        out_zarr,
        joint_dim=len(JOINT_NAMES),
        ee_dim=7,
        cube_dim=0,
        gripper_dim=1,
        obstacle_dim=3,
        flush_every=12,
    )
    writer.set_attrs(
        xml=str(XML_PATH_MULTICUBE),
        joint_names=list(JOINT_NAMES),
        cube_colors=list(CUBE_COLORS),
        state_joints_spec="qpos(joints)",
        state_ee_spec="ee_pos(3) + ee_quat_wxyz(4)",
        state_cube_spec="not_stored_in_multicube_raw",
        state_goal_spec="one_hot(red, green, blue) = 3",
        goal_pos_spec="bin_center_world_xyz(3)",
        state_gripper_spec="gripper_angle(1)",
        action_gripper_spec="gripper_ctrl(1)",
        control_hz=10.0,
        cameras_display=list(CAMERA_NAMES),
    )

    if not args.headless:
        cv2.namedWindow("DAgger Multicube Eval", cv2.WINDOW_AUTOSIZE)

    successes = 0
    total_takeover_steps = 0
    per_color: dict[str, dict[str, int]] = {c: {"success": 0, "total": 0} for c in CUBE_COLORS}

    try:
        ep = 0
        while ep < args.num_episodes:
            ep += 1
            goal = goal_schedule[ep - 1]
            env.set_goal(goal)
            print(f"\n═══ DAgger Episode {ep}/{args.num_episodes}  (goal: {goal}) ═══")
            print("  Policy is running. Press your 'record' key to take over control.")

            success, n_takeover, aborted, replay = run_dagger_episode(
                env,
                model,
                normalizer,
                state_keys,
                action_keys,
                device,
                writer,
                key_to_action,
                max_steps=args.max_steps,
                successes=successes,
                total=ep - 1,
                headless=args.headless,
            )

            if aborted:
                print("Aborted by user.")
                break

            if replay:
                print("  Replaying same episode...")
                ep -= 1
                continue

            total_takeover_steps += n_takeover
            if success:
                successes += 1
            per_color[goal]["total"] += 1
            if success:
                per_color[goal]["success"] += 1

            rate = successes / ep * 100
            result = "SUCCESS" if success else "FAIL"
            print(f"Episode {ep}: {result} | takeover steps this ep: {n_takeover}")
            print(f"  Success rate: {successes}/{ep} ({rate:.0f}%)")

    finally:
        writer.flush()
        cv2.destroyAllWindows()

    n_eps = writer.num_episodes
    n_steps = writer.num_steps_total
    rate = successes / max(1, args.num_episodes) * 100
    print(f"\n{'=' * 50}")
    print("DAgger session complete.")
    print(f"  Episodes evaluated: {args.num_episodes}")
    print(f"  Success rate: {successes}/{args.num_episodes} ({rate:.0f}%)")
    print(f"  Total takeover steps: {total_takeover_steps}")
    print(f"  DAgger episodes saved: {n_eps} ({n_steps} total steps)")
    print(f"  Data saved to: {out_zarr}")
    print(f"\n{'─' * 50}")
    for c in CUBE_COLORS:
        s = per_color[c]["success"]
        t = per_color[c]["total"]
        r = s / t * 100 if t > 0 else 0
        print(f"  {c:6s}: {s}/{t} ({r:.0f}%)")
    print("\n If you collected data, you can now retrain your model with the additional episodes.")


if __name__ == "__main__":
    main()
