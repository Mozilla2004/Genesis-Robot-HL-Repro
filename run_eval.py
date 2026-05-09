"""Run evaluation script for Genesis Robot HL Repro."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
except ImportError:
    print("❌ gymnasium not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Add policies to path
sys.path.insert(0, str(Path(__file__).parent))

from policies.random_policy import RandomPolicy
from policies.halfcheetah_cpg_pd import HalfCheetahCPGPDPolicy


def get_failure_hint(episode_return, steps, max_steps, action_abs_mean=None):
    """
    Generate simple failure hint for next iteration.

    Args:
        episode_return: Total reward
        steps: Number of steps taken
        max_steps: Maximum steps allowed
        action_abs_mean: Mean absolute action value (optional)

    Returns:
        hint: String describing failure mode
    """
    if steps < max_steps * 0.3:
        return "early_termination_or_unstable"
    elif episode_return < 0:
        return "moving_backward_or_high_cost"
    elif action_abs_mean is not None and action_abs_mean < 0.05:
        return "action_too_weak"
    elif steps >= max_steps * 0.8 and episode_return < 100:
        return "survives_but_poor_forward_progress"
    elif episode_return > 200:
        return "working_candidate"
    else:
        return "unknown"


def run_trial(env_id, policy, seed, episodes, max_steps, trial_name, params):
    """
    Run a full trial with multiple episodes.

    Args:
        env_id: Environment ID
        policy: Policy instance
        seed: Random seed
        episodes: Number of episodes
        max_steps: Max steps per episode (None for env default)
        trial_name: Trial identifier
        params: Policy parameters dict

    Returns:
        results: List of episode results
    """
    results = []

    for episode in range(episodes):
        try:
            # Create environment
            env = gym.make(env_id, render_mode=None)
            env.action_space.seed(seed + episode)

            episode_return = 0.0
            steps = 0
            done = False
            truncated = False

            # Track action statistics
            actions_list = []

            obs, info = env.reset(seed=seed + episode)
            policy.reset()

            actual_max_steps = max_steps or 1000  # Default if not specified

            while not (done or truncated) and steps < actual_max_steps:
                action = policy.act(obs)
                actions_list.append(action.copy())
                obs, reward, done, truncated, info = env.step(action)
                episode_return += reward
                steps += 1

            # Calculate action statistics
            actions_array = np.array(actions_list)
            action_abs_mean = float(np.mean(np.abs(actions_array)))
            action_min = float(np.min(actions_array))
            action_max = float(np.max(actions_array))

            # Get environment max steps for failure hint
            if hasattr(env, '_max_episode_steps'):
                env_max_steps = env._max_episode_steps
            else:
                env_max_steps = steps

            # Generate failure hint with action statistics
            failure_hint = get_failure_hint(episode_return, steps, env_max_steps, action_abs_mean)

            result = {
                'episode': episode,
                'return': float(episode_return),
                'steps': int(steps),
                'action_abs_mean': action_abs_mean,
                'action_min': action_min,
                'action_max': action_max,
                'failure_hint': failure_hint
            }
            results.append(result)

            env.close()

        except Exception as e:
            print(f"❌ Episode {episode} failed: {e}")
            results.append({
                'episode': episode,
                'return': 0.0,
                'steps': 0,
                'failure_hint': f'error: {str(e)}'
            })

    return results


def save_results(trial_name, env_id, actual_env_id, policy_name, seed, episodes, results, params):
    """
    Save results to trials.jsonl and summary.csv.

    Args:
        trial_name: Trial identifier
        env_id: Requested environment ID
        actual_env_id: Actual environment ID used
        policy_name: Policy name
        seed: Random seed
        episodes: Number of episodes
        results: List of episode results
        params: Policy parameters dict
    """
    # Ensure runs directory exists
    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)

    trials_jsonl = runs_dir / 'trials.jsonl'
    summary_csv = runs_dir / 'summary.csv'

    timestamp = datetime.now().isoformat()

    # Append to trials.jsonl
    for result in results:
        trial_record = {
            'timestamp': timestamp,
            'trial_name': trial_name,
            'env_id': env_id,
            'actual_env_id': actual_env_id,
            'policy': policy_name,
            'seed': seed,
            'episode': result['episode'],
            'return': result['return'],
            'steps': result['steps'],
            'action_abs_mean': result.get('action_abs_mean', 0.0),
            'action_min': result.get('action_min', 0.0),
            'action_max': result.get('action_max', 0.0),
            'params': params,
            'notes': '',
            'failure_hint': result.get('failure_hint', 'unknown')
        }

        with open(trials_jsonl, 'a') as f:
            f.write(json.dumps(trial_record) + '\n')

    # Calculate summary stats
    returns = [r['return'] for r in results]
    steps_list = [r['steps'] for r in results]
    action_abs_means = [r.get('action_abs_mean', 0.0) for r in results]
    action_mins = [r.get('action_min', 0.0) for r in results]
    action_maxs = [r.get('action_max', 0.0) for r in results]

    summary_record = {
        'trial_name': trial_name,
        'env_id': env_id,
        'actual_env_id': actual_env_id,
        'policy': policy_name,
        'seed': seed,
        'episodes': episodes,
        'mean_return': float(np.mean(returns)),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'mean_steps': float(np.mean(steps_list)),
        'mean_action_abs': float(np.mean(action_abs_means)),
        'min_action': float(np.mean(action_mins)),
        'max_action': float(np.mean(action_maxs)),
        'params_json': json.dumps(params)
    }

    # Append to summary.csv
    if summary_csv.exists():
        df_existing = pd.read_csv(summary_csv)
        df_new = pd.DataFrame([summary_record])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(summary_csv, index=False)
    else:
        df_new = pd.DataFrame([summary_record])
        df_new.to_csv(summary_csv, index=False)

    print(f"\n✅ Results saved:")
    print(f"   - {trials_jsonl}")
    print(f"   - {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description='Run HalfCheetah evaluation')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5', help='Environment ID')
    parser.add_argument('--policy', type=str, choices=['random', 'cpg_pd'], default='random', help='Policy name')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--trial-name', type=str, default=None, help='Trial name')
    parser.add_argument('--max-steps', type=int, default=None, help='Max steps per episode')
    parser.add_argument('--render-video', action='store_true', help='Render video (TODO)')
    parser.add_argument('--params-json', type=str, default=None, help='JSON string for CPG params')

    args = parser.parse_args()

    # Generate trial name if not provided
    if args.trial_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.trial_name = f"{args.policy}_{timestamp}"

    # Parse CPG parameters
    params = {}
    if args.params_json:
        try:
            params = json.loads(args.params_json)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --params-json: {e}")
            return

    # Try to create environment (fallback v5 -> v4)
    env_id = args.env
    actual_env_id = env_id

    try:
        test_env = gym.make(env_id)
        test_env.close()
    except Exception as e:
        print(f"⚠️  {env_id} not available: {e}")
        if env_id == 'HalfCheetah-v5':
            fallback_id = 'HalfCheetah-v4'
            print(f"   Falling back to {fallback_id}")
            try:
                test_env = gym.make(fallback_id)
                test_env.close()
                actual_env_id = fallback_id
            except Exception as e2:
                print(f"❌ {fallback_id} also unavailable: {e2}")
                print("   Please install MuJoCo: pip install -r requirements.txt")
                return
        else:
            print(f"❌ Environment {env_id} not found")
            return

    print(f"\n🏃 Running trial: {args.trial_name}")
    print(f"   Environment: {env_id} -> {actual_env_id}")
    print(f"   Policy: {args.policy}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Seed: {args.seed}")

    # Create policy
    try:
        if args.policy == 'random':
            test_env = gym.make(actual_env_id)
            policy = RandomPolicy(test_env)
            test_env.close()
        elif args.policy == 'cpg_pd':
            test_env = gym.make(actual_env_id)
            policy = HalfCheetahCPGPDPolicy(test_env, **params)
            test_env.close()
        else:
            print(f"❌ Unknown policy: {args.policy}")
            return
    except Exception as e:
        print(f"❌ Failed to create policy: {e}")
        return

    # Run trial
    try:
        results = run_trial(
            env_id=actual_env_id,
            policy=policy,
            seed=args.seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            trial_name=args.trial_name,
            params=params
        )

        # Print summary
        returns = [r['return'] for r in results]
        steps_list = [r['steps'] for r in results]

        print(f"\n📊 Results:")
        print(f"   Mean Return: {np.mean(returns):.2f}")
        print(f"   Min/Max Return: {np.min(returns):.2f} / {np.max(returns):.2f}")
        print(f"   Mean Steps: {np.mean(steps_list):.1f}")

        # Show failure hints
        print(f"\n💡 Failure Hints:")
        for r in results:
            print(f"   Episode {r['episode']}: {r.get('failure_hint', 'unknown')} (return={r['return']:.1f})")

        # Save results
        save_results(
            trial_name=args.trial_name,
            env_id=args.env,
            actual_env_id=actual_env_id,
            policy_name=args.policy,
            seed=args.seed,
            episodes=args.episodes,
            results=results,
            params=params
        )

    except Exception as e:
        print(f"❌ Trial failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
