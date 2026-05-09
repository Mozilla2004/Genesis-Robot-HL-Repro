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
from policies.halfcheetah_residual_mpc import HalfCheetahResidualMPCPolicy


def _extract_x_position(env, obs, info):
    """
    Extract x position from environment (v0.3新增辅助函数).

    Args:
        env: Gymnasium environment
        obs: Current observation
        info: Current info dict

    Returns:
        x_position: X position value or None if not available

    Note:
        v0.3.1: 删除 obs[0] fallback，因为 HalfCheetah observation 通常不包含绝对 x position
    """
    try:
        # 优先从info中读取x_position
        if hasattr(info, 'get') and 'x_position' in info:
            return float(info['x_position'])

        # 尝试从env.unwrapped.data.qpos读取（MuJoCo特定）
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'data'):
            if hasattr(env.unwrapped.data, 'qpos') and len(env.unwrapped.data.qpos) > 0:
                return float(env.unwrapped.data.qpos[0])

    except (IndexError, TypeError, AttributeError) as e:
        # 任何错误都返回None，不要崩溃
        pass

    return None


def get_failure_hint(episode_return, steps, max_steps, action_abs_mean=None, x_displacement=None, mean_x_velocity=None, user_max_steps=None):
    """
    Generate simple failure hint for next iteration.

    Args:
        episode_return: Total reward
        steps: Number of steps taken
        max_steps: Maximum steps allowed (from environment)
        action_abs_mean: Mean absolute action value (optional)
        x_displacement: X position displacement (optional, v0.3新增)
        mean_x_velocity: Mean X velocity (optional, v0.3新增)
        user_max_steps: User-specified max steps (v0.4.6新增)

    Returns:
        hint: String describing failure mode

    Note:
        v0.3.2: 细化 x_displacement 相关提示，提供更精确的诊断
        v0.4.5: 修复 max_steps 相关的误判问题
        v0.4.6: 区分环境默认 max_steps 和用户指定的 user_max_steps
    """
    # v0.4.6: 优先处理用户指定的 max_steps
    if user_max_steps is not None and steps >= user_max_steps:
        # 用户指定 max_steps，根据 x_displacement 细分
        if x_displacement is not None and x_displacement > 0.3:
            return "short_horizon_forward_progress"
        elif x_displacement is not None and x_displacement > 0.05:
            return "short_horizon_weak_forward_motion"
        else:
            return "max_steps_reached_low_progress"

    # v0.4.6: 只有真正提前终止（步数少于环境默认max_steps的30%）才标为不稳定
    # 如果用户没有指定 user_max_steps，使用环境默认的 max_steps 作为参考
    effective_max_steps = user_max_steps if user_max_steps is not None else max_steps
    if effective_max_steps is not None and steps < effective_max_steps * 0.3:
        return "early_termination_or_unstable"

    # v0.3.2改进：细化的failure_hint逻辑，重点关注 x_displacement
    if x_displacement is not None and x_displacement < -0.5:
        return "moving_backward"
    elif x_displacement is not None and x_displacement > 0.5 and episode_return < 100:
        return "forward_but_low_reward"
    elif x_displacement is not None and 0.1 < x_displacement <= 0.5:
        return "weak_forward_motion"
    elif x_displacement is not None and abs(x_displacement) <= 0.1 and action_abs_mean is not None and action_abs_mean < 0.08:
        return "weak_motion_no_forward_progress"
    elif x_displacement is not None and abs(x_displacement) <= 0.1:
        return "no_forward_displacement"
    elif episode_return < -100:
        return "moving_backward_or_high_cost"
    elif episode_return >= 100 or (mean_x_velocity is not None and mean_x_velocity > 0.0008):
        return "working_candidate"
    else:
        return "survives_but_poor_forward_progress"


def run_trial(env_id, policy_name, seed, episodes, max_steps, trial_name, params, user_max_steps=None):
    """
    Run a full trial with multiple episodes.

    Args:
        env_id: Environment ID
        policy_name: Policy name ('random', 'cpg_pd', or 'mpc')
        seed: Random seed
        episodes: Number of episodes
        max_steps: Max steps per episode (None for env default) - DEPRECATED, use user_max_steps
        trial_name: Trial identifier
        params: Policy parameters dict
        user_max_steps: User-specified max steps (v0.4.6)

    Returns:
        results: List of episode results

    Note:
        v0.4.1 hotfix: Policy is now created INSIDE each episode with the real env,
        not beforehand with test_env. This fixes the issue where MPC policy internal
        rollouts needed the actual episode environment, not a closed test_env.
        v0.4.6: Added user_max_steps parameter to distinguish user-specified vs env default.
    """
    results = []

    for episode in range(episodes):
        try:
            import time
            episode_start_time = time.time()

            # Create environment
            env = gym.make(env_id, render_mode=None)
            env.action_space.seed(seed + episode)

            # Create policy instance INSIDE episode with real env (v0.4.1 hotfix)
            # This fixes the issue where MPC internal rollouts need the actual episode env
            if policy_name == 'random':
                policy = RandomPolicy(env)
            elif policy_name == 'cpg_pd':
                policy = HalfCheetahCPGPDPolicy(env, **params)
            elif policy_name == 'mpc':
                policy = HalfCheetahResidualMPCPolicy(env, **params)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            # Check if MPC policy for diagnostics
            is_mpc = hasattr(policy, 'get_diagnostics')

            episode_return = 0.0
            steps = 0
            done = False
            truncated = False

            # Track action statistics
            actions_list = []

            # v0.3新增：x position tracking
            x_position_start = None
            x_position_end = None
            x_displacement = None
            mean_x_velocity = None

            obs, info = env.reset(seed=seed + episode)
            policy.reset()

            # 尝试记录初始x位置
            x_position_start = _extract_x_position(env, obs, info)

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

            # v0.3新增：计算x displacement相关指标
            x_position_end = _extract_x_position(env, obs, info)
            if x_position_start is not None and x_position_end is not None:
                x_displacement = float(x_position_end - x_position_start)
                # 注意：mean_x_velocity 是 displacement per environment step，不一定是物理 m/s
                # v0.3.1: 明确语义，避免误解为物理速度
                mean_x_velocity = float(x_displacement / steps) if steps > 0 else 0.0

            # Get environment max steps for failure hint
            if hasattr(env, '_max_episode_steps'):
                env_max_steps = env._max_episode_steps
            else:
                env_max_steps = steps

            # v0.3改进：生成增强的failure_hint，加入x_displacement和mean_x_velocity
            # v0.4.6: 传递 user_max_steps 到 get_failure_hint
            failure_hint = get_failure_hint(episode_return, steps, env_max_steps, action_abs_mean, x_displacement, mean_x_velocity, user_max_steps)

            # Calculate wall time
            wall_time_sec = time.time() - episode_start_time

            result = {
                'episode': episode,
                'return': float(episode_return),
                'steps': int(steps),
                'action_abs_mean': action_abs_mean,
                'action_min': action_min,
                'action_max': action_max,
                'failure_hint': failure_hint,
                # v0.3新增：x displacement相关字段
                'x_position_start': float(x_position_start) if x_position_start is not None else None,
                'x_position_end': float(x_position_end) if x_position_end is not None else None,
                'x_displacement': x_displacement,
                'mean_x_velocity': mean_x_velocity,
                # v0.4新增：wall time
                'wall_time_sec': wall_time_sec
            }

            # v0.4新增：MPC诊断信息
            if is_mpc:
                try:
                    mpc_diagnostics = policy.get_diagnostics()
                    result.update(mpc_diagnostics)
                except Exception as e:
                    print(f"Warning: Could not get MPC diagnostics: {e}")

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


def save_results(trial_name, env_id, actual_env_id, policy_name, seed, episodes, results, params, user_max_steps=None):
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
        user_max_steps: User-specified max steps (v0.4.6)
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
            'failure_hint': result.get('failure_hint', 'unknown'),
            # v0.3新增：x displacement相关字段
            'x_position_start': result.get('x_position_start'),
            'x_position_end': result.get('x_position_end'),
            'x_displacement': result.get('x_displacement'),
            'mean_x_velocity': result.get('mean_x_velocity'),
            # v0.4新增：wall time
            'wall_time_sec': result.get('wall_time_sec'),
            # v0.4.6新增：user_max_steps
            'user_max_steps': user_max_steps
        }

        # v0.4新增：MPC诊断字段
        mpc_diagnostic_fields = [
            'mpc_horizon', 'mpc_num_candidates', 'mpc_residual_scale',
            'mpc_mean_best_score', 'mpc_mean_score_margin', 'mpc_planning_calls',
            'mpc_mean_planning_time', 'mpc_total_planning_time'
        ]
        for field in mpc_diagnostic_fields:
            if field in result:
                trial_record[field] = result[field]

        with open(trials_jsonl, 'a') as f:
            f.write(json.dumps(trial_record) + '\n')

    # Calculate summary stats
    returns = [r['return'] for r in results]
    steps_list = [r['steps'] for r in results]
    action_abs_means = [r.get('action_abs_mean', 0.0) for r in results]
    action_mins = [r.get('action_min', 0.0) for r in results]
    action_maxs = [r.get('action_max', 0.0) for r in results]

    # v0.3新增：x displacement相关统计
    x_displacements = [r.get('x_displacement') for r in results if r.get('x_displacement') is not None]
    x_velocities = [r.get('mean_x_velocity') for r in results if r.get('mean_x_velocity') is not None]

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
        # v0.3.1修正：使用真正的 min/max，而不是 mean
        'min_action': float(np.min(action_mins)),
        'max_action': float(np.max(action_maxs)),
        # v0.3新增：x displacement相关统计
        'mean_x_displacement': float(np.mean(x_displacements)) if x_displacements else None,
        'mean_x_velocity': float(np.mean(x_velocities)) if x_velocities else None,
        'params_json': json.dumps(params),
        # v0.4.6新增：user_max_steps
        'user_max_steps': user_max_steps
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
    parser.add_argument('--policy', type=str, choices=['random', 'cpg_pd', 'mpc'], default='random', help='Policy name')
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

    # v0.4.1 hotfix: No longer create policy here - policy is created inside run_trial
    # for each episode with the real environment, not with a temporary test_env

    # Run trial (policy creation moved inside run_trial)
    try:
        results = run_trial(
            env_id=actual_env_id,
            policy_name=args.policy,
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
            params=params,
            user_max_steps=args.max_steps  # v0.4.6: Pass user_max_steps
        )

    except Exception as e:
        print(f"❌ Trial failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
