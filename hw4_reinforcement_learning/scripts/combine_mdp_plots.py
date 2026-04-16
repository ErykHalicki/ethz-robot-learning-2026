import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append(str(Path(__file__).resolve().parents[1]))

from envs.grid_world import CliffWalkingEnv
from exercises.ex1_mdp import PolicyIteration, ValueIteration
from scripts.ex1_plot import plot_value_function, plot_policy

LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "mdp"
SLIP_VALUES = [0.0, 0.01, 0.2]
ALGORITHMS = [
    ("Policy Iteration", PolicyIteration),
    ("Value Iteration", ValueIteration),
]


def run_all():
    for algo_name, algo_cls in ALGORITHMS:
        for slip in SLIP_VALUES:
            env = CliffWalkingEnv(slip_chance=slip)
            agent = algo_cls(env, theta=1e-3, gamma=0.9)
            if algo_name == "Policy Iteration":
                value_fn, policy = agent.policy_iteration()
                prefix = "policy_iteration"
            else:
                value_fn, policy = agent.value_iteration()
                prefix = "value_iteration"

            slip_str = f"{slip:.2f}".rstrip("0").rstrip(".")
            plot_value_function(
                env, value_fn,
                title=f"{algo_name} (slip={slip_str}): State Values",
                save_path=LOG_DIR / f"{prefix}_values_slip_{slip_str}.png",
            )
            plot_policy(
                env, policy,
                title=f"{algo_name} (slip={slip_str}): Optimal Policy",
                save_path=LOG_DIR / f"{prefix}_policy_slip_{slip_str}.png",
            )
            plt.close("all")


def make_combined(plot_type, out_name):
    fig, axes = plt.subplots(2, 3, figsize=(36, 8))

    for row, (algo_name, _) in enumerate(ALGORITHMS):
        prefix = "policy_iteration" if row == 0 else "value_iteration"
        for col, slip in enumerate(SLIP_VALUES):
            slip_str = f"{slip:.2f}".rstrip("0").rstrip(".")
            img_path = LOG_DIR / f"{prefix}_{plot_type}_slip_{slip_str}.png"
            img = mpimg.imread(str(img_path))
            axes[row, col].imshow(img)
            axes[row, col].axis("off")

    for col, slip in enumerate(SLIP_VALUES):
        slip_str = f"{slip:.2f}".rstrip("0").rstrip(".")
        axes[0, col].set_title(f"slip = {slip_str}", fontsize=16, pad=10)

    for row, (algo_name, _) in enumerate(ALGORITHMS):
        axes[row, 0].set_ylabel(algo_name, fontsize=16, rotation=90, labelpad=20)
        axes[row, 0].yaxis.set_visible(True)

    kind = "State Values" if plot_type == "values" else "Optimal Policy"
    fig.suptitle(f"{kind}: Policy Iteration vs Value Iteration", fontsize=20, y=1.02)
    fig.tight_layout()
    save_path = LOG_DIR / out_name
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_all()
    make_combined("values", "combined_state_values.png")
    make_combined("policy", "combined_optimal_policy.png")
