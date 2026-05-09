"""HalfCheetah Residual MPC Policy for Genesis Robot HL Repro v0.4.

Lightweight residual MPC on top of stable CPG/PD base gait.
No neural networks, GPU-free, CPU-only implementation.
"""

import numpy as np
import time
import copy

try:
    import gymnasium as gym
except ImportError:
    import gym as gym

try:
    import mujoco
except ImportError:
    mujoco = None  # v0.4.4: mujoco is optional but required for internal rollouts

from .halfcheetah_cpg_pd import HalfCheetahCPGPDPolicy


class HalfCheetahResidualMPCPolicy:
    """Residual MPC policy that improves CPG/PD base gait with short-horizon search.

    This policy:
    1. Uses a stable CPG/PD policy as base
    2. Generates residual action sequences
    3. Scores candidates via short rollouts in copied MuJoCo state
    4. Executes best action and repeats each step
    """

    def __init__(self, env, **params):
        """Initialize residual MPC policy.

        Args:
            env: Gymnasium environment
            **params: Keyword arguments for MPC and base CPG parameters
        """
        self.env = env
        self.params = params or {}

        # Extract base CPG parameters
        base_cpg_params = {k: v for k, v in self.params.items() if k in [
            "gait_type", "direction_sign", "phase_speed", "action_scale",
            "hip_amp", "knee_amp", "ankle_amp", "pitch_gain", "damping_gain"
        ]}

        # Set default base CPG parameters (ft_015 from v0.3.5)
        default_base_params = {
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
        default_base_params.update(base_cpg_params)

        # Initialize base CPG policy
        self.base_policy = HalfCheetahCPGPDPolicy(env, **default_base_params)

        # MPC parameters (conservative for CPU)
        self.horizon = self.params.get("horizon", 4)
        self.num_candidates = self.params.get("num_candidates", 16)
        self.residual_scale = self.params.get("residual_scale", 0.05)
        self.action_smoothness_weight = self.params.get("action_smoothness_weight", 0.02)
        self.control_cost_weight = self.params.get("control_cost_weight", 0.01)
        self.forward_weight = self.params.get("forward_weight", 1.0)
        self.stability_weight = self.params.get("stability_weight", 0.1)
        self.warm_start = self.params.get("warm_start", True)

        # Random number generator for residual candidates
        self.rng = np.random.RandomState(0)

        # v0.4.4: Get MuJoCo model/data for internal rollouts
        self.model = None
        self.data = None
        try:
            unwrapped = env.unwrapped
            self.model = unwrapped.model
            self.data = unwrapped.data
        except Exception as e:
            print(f"Warning: Could not access MuJoCo model/data: {e}")

        # v0.4.4: Internal rollout type for diagnostics
        self.internal_rollout_type = "mujoco_mj_step" if self.model is not None else "env_step_fallback"

        # Timing and diagnostics
        self.planning_calls = 0
        self.total_planning_time = 0.0
        self.last_best_score = None
        self.last_score_margin = None

        # Base CPG policy state
        self.base_policy_state = None

    def reset(self):
        """Reset policy state at episode start."""
        self.base_policy.reset()
        self.planning_calls = 0
        self.total_planning_time = 0.0
        self.last_best_score = None
        self.last_score_margin = None

        # Save initial base policy state
        self._save_base_policy_state()

    def _save_base_policy_state(self):
        """Save base policy state for later restoration."""
        self.base_policy_state = {
            'phase': float(self.base_policy.phase),  # v0.4.3: phase is float, not array
        }

    def _restore_base_policy_state(self, state=None):
        """Restore base policy to saved state."""
        state = state or self.base_policy_state
        if state and 'phase' in state:
            self.base_policy.phase = float(state['phase'])  # v0.4.3: phase is float, not array

    def _get_base_phase(self):
        """Get current base policy phase (v0.4.3 helper - handles float)."""
        # HalfCheetahCPGPDPolicy.phase is a float, not numpy array
        return float(self.base_policy.phase)

    def _set_base_phase(self, phase):
        """Set base policy phase (v0.4.3 helper - handles float)."""
        # HalfCheetahCPGPDPolicy.phase is a float, not numpy array
        self.base_policy.phase = float(phase)

    def act(self, obs):
        """Compute action using residual MPC.

        Args:
            obs: Environment observation

        Returns:
            action: Computed action
        """
        start_time = time.time()

        # v0.4.2 phase handling: Capture phase BEFORE computing base_action
        phase0 = self._get_base_phase()

        # Get base action from CPG policy (this advances phase to phase1)
        base_action = self.base_policy.act(obs)

        # Capture phase AFTER computing base_action
        phase1 = self._get_base_phase()

        # Save current MuJoCo environment state
        mujoco_state = self._save_mujoco_state()
        if mujoco_state is None:
            # Fallback to base action if state save fails
            # Phase should still advance to phase1 since base_action was computed
            return base_action

        try:
            # Generate and score candidate sequences starting from phase0
            best_action, best_score, score_margin = self._plan_with_candidates(
                obs, base_action, mujoco_state, phase0
            )

            # Update diagnostics
            self.last_best_score = best_score
            self.last_score_margin = score_margin
            self.planning_calls += 1

            planning_time = time.time() - start_time
            self.total_planning_time += planning_time

            return best_action

        finally:
            # Always restore environment state
            self._restore_mujoco_state(mujoco_state)
            # v0.4.2 phase handling: Restore to phase1 (after real step), not phase0
            self._set_base_phase(phase1)

    def _save_mujoco_state(self):
        """Save current MuJoCo environment state (v0.4.4 - includes ctrl).

        Returns:
            dict with qpos, qvel, time, ctrl, or None if save fails
        """
        try:
            unwrapped = self.env.unwrapped
            state = {
                'qpos': unwrapped.data.qpos.copy(),
                'qvel': unwrapped.data.qvel.copy(),
                'time': unwrapped.data.time if hasattr(unwrapped.data, 'time') else 0.0
            }
            # v0.4.4: Save ctrl state if available
            if unwrapped.data.ctrl is not None:
                state['ctrl'] = unwrapped.data.ctrl.copy()
            return state
        except Exception as e:
            print(f"Warning: Could not save MuJoCo state: {e}")
            return None

    def _restore_mujoco_state(self, state):
        """Restore MuJoCo environment state (v0.4.4 - includes ctrl).

        Args:
            state: dict with qpos, qvel, time, ctrl
        """
        try:
            unwrapped = self.env.unwrapped

            # Direct state manipulation (v0.4.4: always use direct method for internal rollouts)
            unwrapped.data.qpos[:] = state['qpos']
            unwrapped.data.qvel[:] = state['qvel']

            # Restore time if available
            if hasattr(unwrapped.data, 'time'):
                unwrapped.data.time = state['time']

            # v0.4.4: Restore ctrl state if available
            if 'ctrl' in state and state['ctrl'] is not None and unwrapped.data.ctrl is not None:
                unwrapped.data.ctrl[:] = state['ctrl']

            # Forward physics to ensure consistency
            if mujoco is not None:
                mujoco.mj_forward(unwrapped.model, unwrapped.data)
            else:
                # Try to import mujoco if not already available
                try:
                    import mujoco as mj_fallback
                    mj_fallback.mj_forward(unwrapped.model, unwrapped.data)
                except:
                    pass  # Skip if mujoco not available

        except Exception as e:
            print(f"Warning: Could not restore MuJoCo state: {e}")

    def _plan_with_candidates(self, obs, base_action, mujoco_state, phase0):
        """Generate and score candidate action sequences.

        Args:
            obs: Current observation
            base_action: Base action from CPG policy
            mujoco_state: Saved environment state
            phase0: Base policy phase to start each candidate from (v0.4.2)

        Returns:
            best_action: Best first action from candidates
            best_score: Score of best candidate
            score_margin: Difference between best and second-best
        """
        candidate_scores = []
        candidate_actions = []

        # Generate candidate sequences
        for i in range(self.num_candidates):
            # Restore state for each candidate
            self._restore_mujoco_state(mujoco_state)
            # v0.4.2: Start each candidate from phase0, not from reset state
            self._set_base_phase(phase0)

            # Generate residual sequence
            residual_sequence = self._generate_residual_sequence()

            # Score this candidate
            score, first_action = self._score_candidate(
                obs, base_action, residual_sequence
            )

            candidate_scores.append(score)
            candidate_actions.append(first_action)

        # Find best candidate
        best_idx = np.argmax(candidate_scores)
        best_score = candidate_scores[best_idx]
        best_action = candidate_actions[best_idx]

        # Calculate score margin
        sorted_scores = sorted(candidate_scores, reverse=True)
        score_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        return best_action, best_score, score_margin

    def _mujoco_step_action(self, action):
        """Execute one step using MuJoCo mj_step (v0.4.4 - avoids wrapper pollution).

        Args:
            action: Action to execute
        """
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCo model/data not available for internal rollout")

        # Set control
        self.data.ctrl[:] = action

        # Step physics
        if mujoco is not None:
            mujoco.mj_step(self.model, self.data)
        else:
            raise RuntimeError("mujoco library not available")

    def _get_current_obs(self):
        """Get current observation from MuJoCo state (v0.4.4).

        Returns:
            obs: Current observation array
        """
        try:
            # Try to use environment's _get_obs method if available
            return self.env.unwrapped._get_obs()
        except Exception:
            # Fallback: construct observation from qpos/qvel
            # HalfCheetah observation is typically qpos[1:] + qvel
            qpos = self.data.qpos
            qvel = self.data.qvel
            obs = np.concatenate([qpos[1:], qvel]).astype(np.float64)
            return obs

    def _generate_residual_sequence(self):
        """Generate random residual action sequence.

        Returns:
            residuals: array of shape (horizon, action_dim)
        """
        action_dim = self.env.action_space.shape[0]
        residuals = self.rng.randn(self.horizon, action_dim) * self.residual_scale

        # Clip residuals to reasonable range
        max_residual = 0.1  # Conservative residual limit
        residuals = np.clip(residuals, -max_residual, max_residual)

        return residuals

    def _score_candidate(self, obs, base_action, residual_sequence):
        """Score a candidate via short horizon rollout using MuJoCo mj_step (v0.4.4).

        Args:
            obs: Current observation
            base_action: Base action from CPG
            residual_sequence: Residual actions for horizon

        Returns:
            score: Total score
            first_action: First action to execute
        """
        total_control_cost = 0.0
        total_smoothness_cost = 0.0

        actions = []
        prev_action = None

        # Track x displacement
        x_start = self._get_x_position()
        x_final = x_start

        # Current observation for rollout
        rollout_obs = obs

        for h in range(self.horizon):
            # Get base action for this step
            base_action_h = self.base_policy.act(rollout_obs)

            # Add residual
            residual = residual_sequence[h]
            action = base_action_h + residual

            # Clip to action space
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            actions.append(action)

            # Calculate costs
            control_cost = self.control_cost_weight * np.sum(action ** 2)

            if prev_action is not None:
                smoothness_cost = self.action_smoothness_weight * np.sum((action - prev_action) ** 2)
            else:
                smoothness_cost = 0.0

            total_control_cost += control_cost
            total_smoothness_cost += smoothness_cost

            # v0.4.4: Use MuJoCo mj_step instead of env.step to avoid wrapper pollution
            if self.model is not None and self.data is not None:
                # Use internal MuJoCo step
                self._mujoco_step_action(action)
                # Update observation for next step
                rollout_obs = self._get_current_obs()
            else:
                # Fallback to env.step() (should not happen with proper setup)
                rollout_obs, reward, done, truncated, info = self.env.step(action)
                if done or truncated:
                    break

            prev_action = action

            # Track final x position
            x_final = self._get_x_position()

        # Calculate forward progress
        x_displacement = x_final - x_start

        # Simple torso stability penalty (from qpos[2] - torso angle/height)
        torso_penalty = self._calculate_torso_penalty()
        stability_cost = self.stability_weight * torso_penalty

        # v0.4.4: Score based on interpretable metrics only (no Gymnasium reward)
        score = (
            self.forward_weight * x_displacement
            - total_control_cost
            - total_smoothness_cost
            - stability_cost
        )

        return score, actions[0] if actions else base_action

    def _get_x_position(self):
        """Get current x position from environment."""
        try:
            return self.env.unwrapped.data.qpos[0]
        except:
            return 0.0

    def _calculate_torso_penalty(self):
        """Calculate simple torso stability penalty."""
        try:
            qpos = self.env.unwrapped.data.qpos
            # Penalize large torso angle (qpos[2]) or extreme height (qpos[1])
            torso_penalty = abs(qpos[2]) * 0.1  # Small penalty for torso tilt
            return torso_penalty
        except:
            return 0.0

    def get_diagnostics(self):
        """Get MPC diagnostic information (v0.4.4)."""
        mean_planning_time = (
            self.total_planning_time / self.planning_calls
            if self.planning_calls > 0 else 0.0
        )

        return {
            'mpc_horizon': self.horizon,
            'mpc_num_candidates': self.num_candidates,
            'mpc_residual_scale': self.residual_scale,
            'mpc_mean_best_score': self.last_best_score,
            'mpc_mean_score_margin': self.last_score_margin,
            'mpc_planning_calls': self.planning_calls,
            'mpc_mean_planning_time': mean_planning_time,
            'mpc_total_planning_time': self.total_planning_time,
            'mpc_internal_rollout': self.internal_rollout_type  # v0.4.4: internal rollout method
        }