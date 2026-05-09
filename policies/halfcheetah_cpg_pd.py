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

        # v0.3新增: gait pattern参数
        self.gait_type = params.get('gait_type', 'baseline')
        self.direction_sign = params.get('direction_sign', 1.0)

        # Apply parameter overrides
        for key, value in params.items():
            if hasattr(self, key) and key not in ['gait_type', 'direction_sign']:
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

    def _generate_gait_pattern(self):
        """
        Generate base CPG pattern based on gait_type.

        Returns:
            base_pattern: numpy array of shape (action_dim,)
        """
        base_pattern = np.zeros(self.action_dim)

        # 优雅降级：如果action_dim < 6，只处理前action_dim个关节
        n_joints = min(self.action_dim, 6)

        for i in range(n_joints):
            # 假设HalfCheetah动作顺序：[left_hip, left_knee, left_ankle?, right_hip, right_knee, right_ankle?]
            # 每3个关节一组，对应一条腿
            leg = i // 3  # 0 for left (front), 1 for right (rear)
            joint = i % 3  # 0=hip, 1=knee, 2=ankle

            # 基础幅度
            if joint == 0:
                amp = self.hip_amp
            elif joint == 1:
                amp = self.knee_amp
            else:
                amp = self.ankle_amp

            # 根据gait_type计算相位和幅度调整
            phase_offset, amp_scale = self._get_gait_parameters(self.gait_type, leg, joint)

            # 生成周期性信号
            base_pattern[i] = amp * amp_scale * np.sin(self.phase + phase_offset)

        return base_pattern

    def _get_gait_parameters(self, gait_type, leg, joint):
        """
        Get phase offset and amplitude scale for specific gait type.

        Args:
            gait_type: One of 'baseline', 'mirror', 'rear_drive', 'front_drive', 'alternating', 'bound'
            leg: 0 for left (front), 1 for right (rear)
            joint: 0=hip, 1=knee, 2=ankle

        Returns:
            phase_offset: Phase offset in radians
            amp_scale: Amplitude scaling factor
        """
        if gait_type == 'baseline':
            # 默认交替步态
            phase_offset = leg * np.pi
            amp_scale = 1.0

        elif gait_type == 'mirror':
            # 整体反向，测试动作方向是否反了
            phase_offset = leg * np.pi
            amp_scale = -1.0  # 整体反向

        elif gait_type == 'rear_drive':
            # 强化后腿驱动，前腿动作较小
            if leg == 1:  # rear leg (right)
                phase_offset = leg * np.pi
                amp_scale = 1.2  # 后腿更强
            else:  # front leg (left)
                phase_offset = leg * np.pi
                amp_scale = 0.6  # 前腿较弱

        elif gait_type == 'front_drive':
            # 强化前腿动作，后腿动作较小
            if leg == 0:  # front leg (left)
                phase_offset = leg * np.pi
                amp_scale = 1.2  # 前腿更强
            else:  # rear leg (right)
                phase_offset = leg * np.pi
                amp_scale = 0.6  # 后腿较弱

        elif gait_type == 'alternating':
            # 前后腿严格反相
            phase_offset = leg * np.pi
            amp_scale = 1.0

        elif gait_type == 'bound':
            # 前后腿近似同相，但膝/踝有延迟
            if joint == 0:  # hip 同相
                phase_offset = 0.0
            elif joint == 1:  # knee 略微延迟
                phase_offset = 0.2 * np.pi
            else:  # ankle 更大延迟
                phase_offset = 0.4 * np.pi
            amp_scale = 1.0

        else:
            # 未知gait_type，回退到baseline
            phase_offset = leg * np.pi
            amp_scale = 1.0

        return phase_offset, amp_scale

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

        # Generate base pattern based on gait_type
        base_pattern = self._generate_gait_pattern()

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

        # v0.3新增：应用方向符号
        action = action * self.direction_sign

        # Clip to action space bounds
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        return action
