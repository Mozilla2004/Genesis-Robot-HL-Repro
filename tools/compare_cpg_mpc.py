"""CPG vs MPC Comparison Tool for Genesis Robot HL Repro v0.4.5."""

import argparse
import json
import csv
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gymnasium as gym
except ImportError:
    print("❌ gymnasium not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

from run_eval import run_trial, save_results


# Fixed parameters for comparison
CPG_FT015_PARAMS = {
    "gait_type": "alternating",
    "direction_sign": 1.0,
    "phase_speed": 0.35,
    "action_scale": 0.60,
    "hip_amp": 0.50,
    "knee_amp": 0.30,
    "ankle_amp": 0.15,
    "pitch_gain": 0.06,
    "damping_gain": 0.025
}

MPC_004_PARAMS = {
    "horizon": 4,
    "num_candidates": 16,
    "residual_scale": 0.08,
    "forward_weight": 1.0,
    "control_cost_weight": 0.02,
    "action_smoothness_weight": 0.03,
    "stability_weight": 0.1,
    # Include base CPG parameters
    "gait_type": "alternating",
    "direction_sign": 1.0,
    "phase_speed": 0.35,
    "action_scale": 0.60,
    "hip_amp": 0.50,
    "knee_amp": 0.30,
    "ankle_amp": 0.15,
    "pitch_gain": 0.06,
    "damping_gain": 0.025
}


def run_comparison(env_id="HalfCheetah-v5", steps_list=[100, 300], seed=0, clean_first=False):
    """Run CPG vs MPC comparison.

    Args:
        env_id: Environment ID
        steps_list: List of step counts to test
        seed: Random seed
        clean_first: Whether to clean runs directory first
    """
    print(f"🔬 CPG vs MPC Comparison")
    print(f"   Environment: {env_id}")
    print(f"   Steps: {steps_list}")
    print(f"   Seed: {seed}")
    print(f"   Clean first: {clean_first}")

    # Clean runs if requested
    if clean_first:
        print("\n🧹 Cleaning runs directory...")
        try:
            from tools.clean_runs import clean_runs
            clean_runs()
        except ImportError:
            print("  ✗ Failed to import clean_runs")
            subprocess.run([sys.executable, "tools/clean_runs.py"], cwd=Path(__file__).parent.parent)

    comparison_results = []

    # Define test cases
    test_cases = [
        # (policy_name, trial_suffix, params)
        ("cpg_pd", "ft015", CPG_FT015_PARAMS),
        ("mpc", "mpc_004", MPC_004_PARAMS),
    ]

    # Try environment detection
    try:
        test_env = gym.make(env_id)
        test_env.close()
        actual_env_id = env_id
    except Exception as e:
        print(f"⚠️  {env_id} not available: {e}")
        if env_id == 'HalfCheetah-v5':
            actual_env_id = 'HalfCheetah-v4'
            print(f"   Falling back to {actual_env_id}")
        else:
            print(f"❌ Environment {env_id} not found")
            return

    # Run tests
    for steps in steps_list:
        print(f"\n🎯 Running {steps} steps comparison...")

        # Store CPG baseline for this horizon
        cpg_baseline = None

        for policy_name, trial_suffix, params in test_cases:
            trial_name = f"{policy_name}_{trial_suffix}_{steps}steps"

            print(f"\n  📊 Testing: {trial_name}")

            try:
                # Run trial
                results = run_trial(
                    env_id=actual_env_id,
                    policy_name=policy_name,
                    seed=seed,
                    episodes=1,
                    max_steps=steps,
                    trial_name=trial_name,
                    params=params
                )

                # Save individual trial results
                save_results(
                    trial_name=trial_name,
                    env_id=env_id,
                    actual_env_id=actual_env_id,
                    policy_name=policy_name,
                    seed=seed,
                    episodes=1,
                    results=results,
                    params=params,
                    user_max_steps=steps  # v0.4.6: 使用 user_max_steps 参数
                )

                # Extract summary statistics
                if results:
                    result = results[0]  # Single episode

                    # Store CPG baseline if this is the CPG policy
                    if policy_name == "cpg_pd":
                        cpg_baseline = result

                    comparison_row = {
                        "horizon_steps": steps,
                        "policy": f"{policy_name}_{trial_suffix}",
                        "return": result.get('return', 0.0),
                        "x_displacement": result.get('x_displacement', 0.0),
                        "action_abs_mean": result.get('action_abs_mean', 0.0),
                        "wall_time_sec": result.get('wall_time_sec', 0.0),
                        "steps": result.get('steps', 0),
                        "failure_hint": result.get('failure_hint', 'unknown')
                    }

                    # Calculate gains if we have CPG baseline
                    if cpg_baseline and policy_name == "mpc":
                        comparison_row["return_gain_vs_cpg"] = (
                            comparison_row["return"] - cpg_baseline.get('return', 0.0)
                        )
                        comparison_row["displacement_gain_vs_cpg"] = (
                            comparison_row["x_displacement"] - cpg_baseline.get('x_displacement', 0.0)
                        )
                        comparison_row["speedup_or_slowdown"] = (
                            comparison_row["wall_time_sec"] / cpg_baseline.get('wall_time_sec', 1.0)
                            if cpg_baseline.get('wall_time_sec', 0.0) > 0 else 1.0
                        )
                    else:
                        comparison_row["return_gain_vs_cpg"] = 0.0
                        comparison_row["displacement_gain_vs_cpg"] = 0.0
                        comparison_row["speedup_or_slowdown"] = 1.0

                    comparison_results.append(comparison_row)

                    print(f"     ✓ return={result.get('return', 0.0):.2f}, "
                          f"x_disp={result.get('x_displacement', 0.0):.3f}, "
                          f"steps={result.get('steps', 0)}, "
                          f"time={result.get('wall_time_sec', 0.0):.1f}s")

            except Exception as e:
                print(f"     ✗ Failed: {e}")

    # Save comparison results
    if comparison_results:
        output_file = "runs/comparison_results.csv"
        try:
            # Ensure runs directory exists
            Path("runs").mkdir(exist_ok=True)

            # Write CSV
            fieldnames = [
                "horizon_steps", "policy", "return", "x_displacement",
                "action_abs_mean", "wall_time_sec", "steps", "failure_hint",
                "speedup_or_slowdown", "return_gain_vs_cpg", "displacement_gain_vs_cpg"
            ]

            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparison_results)

            print(f"\n✅ Comparison results saved to {output_file}")

            # Print summary
            print(f"\n📊 Comparison Summary:")
            print(f"{'Steps':<10} {'Policy':<15} {'Return':<10} {'X-Disp':<10} {'Gain vs CPG':<15}")
            print("-" * 60)

            current_steps = None
            cpg_result = None

            for row in comparison_results:
                if row["horizon_steps"] != current_steps:
                    current_steps = row["horizon_steps"]
                    cpg_result = None

                if row["policy"] == "cpg_ft015":
                    cpg_result = row
                    print(f"{row['horizon_steps']:<10} {row['policy']:<15} "
                          f"{row['return']:<10.2f} {row['x_displacement']:<10.3f} {'baseline':<15}")
                else:
                    gain_str = f"+{row['return_gain_vs_cpg']:.2f}"
                    disp_gain_str = f"+{row['displacement_gain_vs_cpg']:.3f}"
                    print(f"{row['horizon_steps']:<10} {row['policy']:<15} "
                          f"{row['return']:<10.2f} {row['x_displacement']:<10.3f} "
                          f"{gain_str:>7} / {disp_gain_str:>7}")

        except Exception as e:
            print(f"❌ Failed to save comparison results: {e}")
    else:
        print("❌ No comparison results to save")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare CPG vs MPC performance")
    parser.add_argument("--env", default="HalfCheetah-v5", help="Environment ID")
    parser.add_argument("--steps", default="100,300", help="Comma-separated step counts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--clean", action="store_true", help="Clean runs directory before comparison")

    args = parser.parse_args()

    # Parse steps list
    steps_list = [int(s.strip()) for s in args.steps.split(",")]

    run_comparison(
        env_id=args.env,
        steps_list=steps_list,
        seed=args.seed,
        clean_first=args.clean
    )


if __name__ == "__main__":
    main()