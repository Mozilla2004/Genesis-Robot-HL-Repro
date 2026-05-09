"""CPG + PD heuristic policy for HalfCheetah."""

import numpy as np


class HalfCheetahCPGPDPolicy:
    """
    Heuristic policy using Central Pattern Generator (CPG) + Proportional-Derivative (PD) control.

    No neural networks, just numpy-based periodic gait with simple posture feedback.
    """

    def __init__(self, env, **params):
        """
        Initialize CPG/PD policy.

        Args:
            env: Gymnasium environment
            **params: Optional parameter overrides
        """
        self.action_space = env.action_space
        self.action_dim = self.action_space.shape[0]

        # Default parameters (tuned for basic forward motion)
        self.phase_speed = 0.1
        self.hip_amp = 0.3
        self.knee_amp = 0.2
        self.ankle_amp = 0.1
        self.action_scale = 0.5
        self.pitch_gain = 0.1
        self.damping_gain = 0.05

        # Apply parameter overrides
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Internal state
        self.phase = 0.0

        # Safe observation indices (with bounds checking)
        # NOTE: HalfCheetah observation mapping may vary by version
        # v0.1.1 uses cautious approximation:
        # - obs[0] usually: position/coords (not used for pitch)
        # - obs[1] usually: torso angle/pitch (main feedback signal)
        # - obs后半部分: velocities (for damping estimation)
        obs_dim = env.observation_space.shape[0]
        self.torso_angle_idx = 1 if obs_dim > 1 else None
        self.velocity_start_idx = max(obs_dim // 2, 6)  # 后半段作为速度

    def reset(self):
        """Reset internal phase."""
        self.phase = 0.0

    def act(self, obs):
        """
        Generate action using CPG + PD.

        Args:
            obs: Observation vector

        Returns:
            action: Clipped action vector
        """
        # Update phase
        self.phase += self.phase_speed
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # Generate base pattern (sinusoidal gait)
        base_pattern = np.zeros(self.action_dim)

        # Assume HalfCheetah has 6 actions: [left_hip, left_knee, left_ankle?, right_hip, right_kee, right_ankle?]
        # Use alternating pattern for left/right legs
        for i in range(self.action_dim):
            leg = i // 3  # 0 for left, 1 for right
            joint = i % 3  # 0=hip, 1=knee, 2=ankle

            # Phase offset for alternating gait
            phase_offset = leg * np.pi

            # Amplitude per joint
            if joint == 0:
                amp = self.hip_amp
            elif joint == 1:
                amp = self.knee_amp
            else:
                amp = self.ankle_amp

            # Generate sinusoidal pattern
            base_pattern[i] = amp * np.sin(self.phase + phase_offset)

        # Add simple posture feedback (pitch correction)
        pitch_correction = 0.0
        if self.torso_angle_idx is not None and len(obs) > self.torso_angle_idx:
            try:
                torso_angle = obs[self.torso_angle_idx]
                pitch_correction = -self.pitch_gain * torso_angle
            except (IndexError, TypeError):
                pass

        # Apply damping (方案A: 极简阻尼计算)
        damping_scalar = 0.0
        if self.damping_gain > 0 and len(obs) > self.velocity_start_idx:
            try:
                # 从后半段估计速度部分
                velocity_part = obs[self.velocity_start_idx:]
                # 计算阻尼标量：tanh保持有界，abs考虑速度大小，mean综合
                damping_scalar = self.damping_gain * np.tanh(np.mean(np.abs(velocity_part)))
            except (IndexError, TypeError):
                pass

        # Combine base pattern + feedback
        # damping 作为标量乘以 sign(base_pattern)，使高振幅动作获得更大阻尼
        damping_vector = damping_scalar * np.sign(base_pattern)
        action = base_pattern * self.action_scale + pitch_correction - damping_vector

        # Clip to action space bounds
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        return action
