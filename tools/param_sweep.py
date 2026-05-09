"""Parameter sweep tool for Genesis Robot HL Repro."""

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gymnasium as gym
except ImportError:
    print("❌ gymnasium not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# v0.4.1 hotfix: No longer need to import policy classes here
# Policies are now created inside run_trial with the real environment
from run_eval import run_trial, save_results


# Parameter presets
PRESETS = {
    'tiny': [
        # 快速 smoke test，5 个候选
        {"candidate_id": "tiny_001", "phase_speed": 0.12, "action_scale": 0.8, "hip_amp": 0.5,
         "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.03},
        {"candidate_id": "tiny_002", "phase_speed": 0.16, "action_scale": 1.0, "hip_amp": 0.7,
         "knee_amp": 0.5, "ankle_amp": 0.25, "pitch_gain": 0.05, "damping_gain": 0.03},
        {"candidate_id": "tiny_003", "phase_speed": 0.20, "action_scale": 1.2, "hip_amp": 0.9,
         "knee_amp": 0.6, "ankle_amp": 0.30, "pitch_gain": 0.05, "damping_gain": 0.03},
        {"candidate_id": "tiny_004", "phase_speed": 0.28, "action_scale": 1.0, "hip_amp": 1.0,
         "knee_amp": 0.8, "ankle_amp": 0.40, "pitch_gain": 0.0, "damping_gain": 0.03},
        {"candidate_id": "tiny_005", "phase_speed": 0.16, "action_scale": 1.2, "hip_amp": 0.8,
         "knee_amp": 0.5, "ankle_amp": 0.25, "pitch_gain": 0.0, "damping_gain": 0.0},
    ],

    'small': [
        # 系统扫描，12 个候选
        {"candidate_id": "small_001", "phase_speed": 0.12, "action_scale": 0.8, "hip_amp": 0.5,
         "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.03},
        {"candidate_id": "small_002", "phase_speed": 0.16, "action_scale": 1.0, "hip_amp": 0.7,
         "knee_amp": 0.5, "ankle_amp": 0.25, "pitch_gain": 0.05, "damping_gain": 0.03},
        {"candidate_id": "small_003", "phase_speed": 0.20, "action_scale": 1.0, "hip_amp": 0.9,
         "knee_amp": 0.6, "ankle_amp": 0.30, "pitch_gain": 0.05, "damping_gain": 0.03},
        {"candidate_id": "small_004", "phase_speed": 0.28, "action_scale": 1.0, "hip_amp": 1.0,
         "knee_amp": 0.8, "ankle_amp": 0.40, "pitch_gain": 0.0, "damping_gain": 0.03},
        {"candidate_id": "small_005", "phase_speed": 0.16, "action_scale": 1.2, "hip_amp": 0.8,
         "knee_amp": 0.5, "ankle_amp": 0.25, "pitch_gain": 0.0, "damping_gain": 0.0},
        {"candidate_id": "small_006", "phase_speed": 0.20, "action_scale": 1.2, "hip_amp": 1.0,
         "knee_amp": 0.7, "ankle_amp": 0.35, "pitch_gain": 0.0, "damping_gain": 0.0},
        # 添加更高频率和大振幅的候选
        {"candidate_id": "small_007", "phase_speed": 0.24, "action_scale": 1.0, "hip_amp": 1.1,
         "knee_amp": 0.8, "ankle_amp": 0.40, "pitch_gain": 0.03, "damping_gain": 0.02},
        {"candidate_id": "small_008", "phase_speed": 0.32, "action_scale": 1.1, "hip_amp": 1.2,
         "knee_amp": 0.9, "ankle_amp": 0.45, "pitch_gain": 0.0, "damping_gain": 0.02},
        {"candidate_id": "small_009", "phase_speed": 0.18, "action_scale": 1.3, "hip_amp": 0.9,
         "knee_amp": 0.6, "ankle_amp": 0.30, "pitch_gain": 0.08, "damping_gain": 0.04},
        {"candidate_id": "small_010", "phase_speed": 0.22, "action_scale": 1.3, "hip_amp": 1.1,
         "knee_amp": 0.7, "ankle_amp": 0.35, "pitch_gain": 0.06, "damping_gain": 0.03},
        {"candidate_id": "small_011", "phase_speed": 0.36, "action_scale": 1.2, "hip_amp": 1.3,
         "knee_amp": 1.0, "ankle_amp": 0.50, "pitch_gain": 0.0, "damping_gain": 0.01},
        {"candidate_id": "small_012", "phase_speed": 0.40, "action_scale": 1.4, "hip_amp": 1.4,
         "knee_amp": 1.1, "ankle_amp": 0.55, "pitch_gain": 0.0, "damping_gain": 0.01},
    ],

    # v0.3新增：gait_tiny preset，用于测试gait_type和direction_sign
    'gait_tiny': [
        # 测试不同gait_type，共12个候选
        {"candidate_id": "gait_001", "gait_type": "baseline", "direction_sign": 1.0, "phase_speed": 0.10,
         "action_scale": 0.5, "hip_amp": 0.3, "knee_amp": 0.2, "ankle_amp": 0.1, "pitch_gain": 0.05, "damping_gain": 0.03},

        {"candidate_id": "gait_002", "gait_type": "mirror", "direction_sign": 1.0, "phase_speed": 0.10,
         "action_scale": 0.5, "hip_amp": 0.3, "knee_amp": 0.2, "ankle_amp": 0.1, "pitch_gain": 0.05, "damping_gain": 0.03},

        {"candidate_id": "gait_003", "gait_type": "rear_drive", "direction_sign": 1.0, "phase_speed": 0.12,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_004", "gait_type": "rear_drive", "direction_sign": -1.0, "phase_speed": 0.12,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_005", "gait_type": "front_drive", "direction_sign": 1.0, "phase_speed": 0.12,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_006", "gait_type": "front_drive", "direction_sign": -1.0, "phase_speed": 0.12,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_007", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.12,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_008", "gait_type": "alternating", "direction_sign": -1.0, "phase_speed": 0.12,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_009", "gait_type": "bound", "direction_sign": 1.0, "phase_speed": 0.16,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "gait_010", "gait_type": "bound", "direction_sign": -1.0, "phase_speed": 0.16,
         "action_scale": 0.6, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        # 额外的测试候选：更高频率的rear_drive和alternating
        {"candidate_id": "gait_011", "gait_type": "rear_drive", "direction_sign": 1.0, "phase_speed": 0.18,
         "action_scale": 0.8, "hip_amp": 0.6, "knee_amp": 0.4, "ankle_amp": 0.2, "pitch_gain": 0.02, "damping_gain": 0.01},

        {"candidate_id": "gait_012", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.7, "hip_amp": 0.5, "knee_amp": 0.3, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},
    ],

    # v0.3.2新增：alternating_local preset，围绕gait_012做局部搜索
    'alternating_local': [
        # 基于gait_012的最佳参数进行局部搜索
        # Base: phase_speed=0.20, action_scale=0.70, hip_amp=0.50, knee_amp=0.30, ankle_amp=0.15, pitch_gain=0.04, damping_gain=0.02

        # 1. Baseline: 原始gait_012作为对照
        {"candidate_id": "alt_001", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 2. Phase Speed探索 (A组)
        {"candidate_id": "alt_002", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.16,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "alt_003", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.24,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "alt_004", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.28,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 3. Action Scale探索 (B组)
        {"candidate_id": "alt_005", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.55, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "alt_006", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.85, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 4. Hip/Knee组合探索 (C组)
        {"candidate_id": "alt_007", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.70, "hip_amp": 0.45, "knee_amp": 0.25, "ankle_amp": 0.12, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "alt_008", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.70, "hip_amp": 0.55, "knee_amp": 0.35, "ankle_amp": 0.18, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "alt_009", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.70, "hip_amp": 0.65, "knee_amp": 0.45, "ankle_amp": 0.22, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 5. Pitch/Damping探索 (D组)
        {"candidate_id": "alt_010", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.02, "damping_gain": 0.01},

        {"candidate_id": "alt_011", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.06, "damping_gain": 0.04},

        # 6. 组合优化：最有希望的方向
        {"candidate_id": "alt_012", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.24,
         "action_scale": 0.85, "hip_amp": 0.55, "knee_amp": 0.35, "ankle_amp": 0.18, "pitch_gain": 0.06, "damping_gain": 0.04},

        # 7. 额外探索：更低的damping + 更高的phase_speed
        {"candidate_id": "alt_013", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.24,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.01},

        # 8. 更高的hip/knee驱动
        {"candidate_id": "alt_014", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.20,
         "action_scale": 0.85, "hip_amp": 0.65, "knee_amp": 0.45, "ankle_amp": 0.22, "pitch_gain": 0.02, "damping_gain": 0.01},

        # 9. 平衡方案：中等参数
        {"candidate_id": "alt_015", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.22,
         "action_scale": 0.75, "hip_amp": 0.55, "knee_amp": 0.35, "ankle_amp": 0.18, "pitch_gain": 0.05, "damping_gain": 0.03},

        # 10. 保守方案：降低频率，增加稳定性
        {"candidate_id": "alt_016", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.18,
         "action_scale": 0.65, "hip_amp": 0.45, "knee_amp": 0.25, "ankle_amp": 0.12, "pitch_gain": 0.03, "damping_gain": 0.02},
    ],

    # v0.3.3新增：phase_refine preset，围绕alt_004做phase_speed精细搜索
    'phase_refine': [
        # 基于alt_004的最佳参数：phase_speed=0.28, action_scale=0.70, return=+19.51
        # 固定参数：gait_type="alternating", direction_sign=1.0, hip_amp=0.50, knee_amp=0.30, ankle_amp=0.15

        # 1. Baseline: alt_004原参数作为对照
        {"candidate_id": "ph_001", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.28,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 2. Phase Speed核心探索：0.30/0.32 是否继续提升
        {"candidate_id": "ph_002", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "ph_003", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.32,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "ph_004", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 3. 保守探索：略低phase_speed + 降低action_scale
        {"candidate_id": "ph_005", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.26,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 4. Action Scale探索：在高phase_speed下测试不同action_scale
        {"candidate_id": "ph_006", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "ph_007", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.80, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.03},

        # 5. 更高phase_speed + 保守action_scale
        {"candidate_id": "ph_008", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.32,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        {"candidate_id": "ph_009", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.32,
         "action_scale": 0.80, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.03},

        # 6. Pitch/Damping精细调整
        {"candidate_id": "ph_010", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "ph_011", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.03},

        # 7. 更高phase_speed + Pitch/Damping调整
        {"candidate_id": "ph_012", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.32,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.03, "damping_gain": 0.02},

        {"candidate_id": "ph_013", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.32,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.03},

        # 8. 最优组合推测：高phase_speed + 保守其他参数
        {"candidate_id": "ph_014", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # 9. 边界探索：测试0.36是否过界
        {"candidate_id": "ph_015", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.36,
         "action_scale": 0.65, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.03},

        # 10. 平衡优化：中等phase_speed + 中等action_scale + 中等反馈
        {"candidate_id": "ph_016", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.31,
         "action_scale": 0.75, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.045, "damping_gain": 0.025},
    ],

    # v0.3.4新增：phase_top_regression preset，对v0.3.3的top candidates做多seed回归
    'phase_top_regression': [
        # 基于v0.3.3结果，选择top 6 candidates做稳定性验证
        # 目标：验证ph_014 (return=+35.59, x_displacement=+2.352) 的稳定性

        # 1. 最佳候选：ph_014
        {"candidate_id": "reg_001", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # 2. ph_008：高phase_speed + 保守action_scale
        {"candidate_id": "reg_002", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.32,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 3. ph_006：中等phase_speed + 保守action_scale
        {"candidate_id": "reg_003", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},

        # 4. ph_011：中等phase_speed + 高反馈
        {"candidate_id": "reg_004", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.30,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.03},

        # 5. ph_015：边界探索 - 测试0.36是否过界
        {"candidate_id": "reg_005", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.36,
         "action_scale": 0.65, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.03},

        # 6. ph_001：历史baseline - alt_004原参数
        {"candidate_id": "reg_006", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.28,
         "action_scale": 0.70, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.04, "damping_gain": 0.02},
    ],

    # v0.3.5新增：base_finetune preset，围绕reg_001/ph_014做精细微调
    'base_finetune': [
        # 基于v0.3.4稳定结果：reg_001/ph_014, mean_return=+35.54
        # 目标：通过精细微调把mean_return从35推到50+
        # 固定参数：gait_type="alternating", direction_sign=1.0, hip_amp=0.50, knee_amp=0.30, ankle_amp=0.15

        # ===== 基线（reg_001/ph_014） =====
        # 1. Baseline: v0.3.4的最佳稳定候选
        {"candidate_id": "ft_001", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # ===== Action Scale探索（保守→中等） =====
        # 2. 更保守action_scale，其他不变
        {"candidate_id": "ft_002", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.55, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # 3. 略高action_scale，其他不变
        {"candidate_id": "ft_003", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.65, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # ===== Phase Speed微调 =====
        # 4. 略高phase_speed，其他不变
        {"candidate_id": "ft_004", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.35,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # 5. 略低phase_speed，其他不变
        {"candidate_id": "ft_005", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.33,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # ===== 组合探索：Phase Speed + Action Scale =====
        # 6. 高phase_speed + 低action_scale
        {"candidate_id": "ft_006", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.35,
         "action_scale": 0.55, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # 7. 高phase_speed + 高action_scale
        {"candidate_id": "ft_007", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.35,
         "action_scale": 0.65, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # 8. 低phase_speed + 低action_scale
        {"candidate_id": "ft_008", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.33,
         "action_scale": 0.55, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.02},

        # ===== Pitch Gain微调 =====
        # 9. 略高pitch_gain
        {"candidate_id": "ft_009", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.055, "damping_gain": 0.02},

        # 10. 更高pitch_gain
        {"candidate_id": "ft_010", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.06, "damping_gain": 0.02},

        # 11. 略低pitch_gain
        {"candidate_id": "ft_011", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.045, "damping_gain": 0.02},

        # ===== Damping Gain微调 =====
        # 12. 略低damping_gain
        {"candidate_id": "ft_012", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.015},

        # 13. 略高damping_gain
        {"candidate_id": "ft_013", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.34,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.05, "damping_gain": 0.025},

        # ===== 高级组合 =====
        # 14. 高phase_speed + 高pitch_gain
        {"candidate_id": "ft_014", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.35,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.055, "damping_gain": 0.02},

        # 15. 高phase_speed + 高pitch_gain + 高damping
        {"candidate_id": "ft_015", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.35,
         "action_scale": 0.60, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.06, "damping_gain": 0.025},

        # 16. 平衡优化：中等所有参数
        {"candidate_id": "ft_016", "gait_type": "alternating", "direction_sign": 1.0, "phase_speed": 0.345,
         "action_scale": 0.625, "hip_amp": 0.50, "knee_amp": 0.30, "ankle_amp": 0.15, "pitch_gain": 0.0525, "damping_gain": 0.0225},
    ],

    # v0.4新增：mpc_tiny preset，测试lightweight residual MPC
    'mpc_tiny': [
        # 基于v0.3.5的ft_015作为base gait，加入residual MPC
        # 目标：验证MPC是否可行、是否能提升base gait、CPU成本是否可接受

        # 1. 最保守配置
        {"candidate_id": "mpc_001", "horizon": 3, "num_candidates": 8, "residual_scale": 0.03,
         "forward_weight": 1.0, "control_cost_weight": 0.01, "action_smoothness_weight": 0.02, "stability_weight": 0.1},

        # 2. 轻度提升
        {"candidate_id": "mpc_002", "horizon": 4, "num_candidates": 12, "residual_scale": 0.04,
         "forward_weight": 1.0, "control_cost_weight": 0.01, "action_smoothness_weight": 0.02, "stability_weight": 0.1},

        # 3. 中等配置（基于v0.4默认参数）
        {"candidate_id": "mpc_003", "horizon": 4, "num_candidates": 16, "residual_scale": 0.05,
         "forward_weight": 1.0, "control_cost_weight": 0.01, "action_smoothness_weight": 0.02, "stability_weight": 0.1},

        # 4. 更大residual scale
        {"candidate_id": "mpc_004", "horizon": 4, "num_candidates": 16, "residual_scale": 0.08,
         "forward_weight": 1.0, "control_cost_weight": 0.02, "action_smoothness_weight": 0.03, "stability_weight": 0.1},

        # 5. 更长horizon + 更高forward weight
        {"candidate_id": "mpc_005", "horizon": 5, "num_candidates": 16, "residual_scale": 0.05,
         "forward_weight": 1.2, "control_cost_weight": 0.01, "action_smoothness_weight": 0.02, "stability_weight": 0.1},

        # 6. 更长horizon + 更多candidates
        {"candidate_id": "mpc_006", "horizon": 5, "num_candidates": 24, "residual_scale": 0.05,
         "forward_weight": 1.2, "control_cost_weight": 0.01, "action_smoothness_weight": 0.02, "stability_weight": 0.15},
    ]
}


def run_param_sweep(env_id, preset_name, episodes_per_seed, seeds, max_steps, policy_type='cpg_pd'):
    """
    Run parameter sweep across candidates.

    Args:
        env_id: Environment ID
        preset_name: Preset name ('tiny' or 'small')
        episodes_per_seed: Episodes per seed
        seeds: List of seeds or comma-separated string
        max_steps: Max steps per episode
        policy_type: Policy type ('cpg_pd' or 'mpc')

    Returns:
        sweep_results: List of sweep results
    """
    # Parse seeds
    if isinstance(seeds, str):
        seeds = [int(s) for s in seeds.split(',')]

    # Get preset candidates
    candidates = PRESETS.get(preset_name, [])
    if not candidates:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"   Available presets: {list(PRESETS.keys())}")
        return []

    print(f"\n🔬 Running {preset_name} parameter sweep:")
    print(f"   Candidates: {len(candidates)}")
    print(f"   Seeds: {seeds}")
    print(f"   Episodes per seed: {episodes_per_seed}")
    print(f"   Total trials: {len(candidates) * len(seeds)}")
    print(f"   Policy type: {policy_type}")

    sweep_results = []

    # Try to create environment once to verify
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
            return []

    # Run each candidate
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        params = {k: v for k, v in candidate.items() if k != "candidate_id"}

        print(f"\n🎯 Testing candidate: {candidate_id}")
        print(f"   Params: {json.dumps(params, indent=2)}")

        # Aggregate results across seeds
        all_returns = []
        all_steps = []
        all_action_abs = []
        all_failure_hints = []
        # v0.3新增：x displacement相关统计
        all_x_displacements = []
        all_x_velocities = []
        # v0.4新增：wall time统计
        all_wall_times = []

        # Run across all seeds
        for seed in seeds:
            trial_name = f"sweep_{preset_name}_{candidate_id}_seed{seed}"

            try:
                # v0.4.1 hotfix: No longer create policy here - policy is created inside run_trial
                # for each episode with the real environment, not with a temporary test_env

                # Determine policy name based on policy type
                if policy_type == 'mpc':
                    policy_name = 'mpc'
                else:  # cpg_pd
                    policy_name = 'cpg_pd'

                # Run trial (policy creation moved inside run_trial)
                results = run_trial(
                    env_id=actual_env_id,
                    policy_name=policy_name,
                    seed=seed,
                    episodes=episodes_per_seed,
                    max_steps=max_steps,
                    trial_name=trial_name,
                    params=params
                )

                # Aggregate statistics
                for result in results:
                    all_returns.append(result['return'])
                    all_steps.append(result['steps'])
                    all_action_abs.append(result.get('action_abs_mean', 0.0))
                    all_failure_hints.append(result.get('failure_hint', 'unknown'))
                    # v0.3新增：收集x displacement相关字段
                    if result.get('x_displacement') is not None:
                        all_x_displacements.append(result['x_displacement'])
                    if result.get('mean_x_velocity') is not None:
                        all_x_velocities.append(result['mean_x_velocity'])
                    # v0.4新增：收集wall time
                    if result.get('wall_time_sec') is not None:
                        all_wall_times.append(result['wall_time_sec'])

                # Save individual trial results
                save_results(
                    trial_name=trial_name,
                    env_id=env_id,
                    actual_env_id=actual_env_id,
                    policy_name=policy_name,
                    seed=seed,
                    episodes=episodes_per_seed,
                    results=results,
                    params=params
                )

                # Print seed-level summary
                seed_returns = [r['return'] for r in results]
                print(f"   Seed {seed}: mean_return={np.mean(seed_returns):.2f}")

            except Exception as e:
                print(f"   ❌ Seed {seed} failed: {e}")

        # Calculate candidate-level statistics
        if all_returns:
            candidate_result = {
                'candidate_id': candidate_id,
                'preset': preset_name,
                'seeds': ",".join(map(str, seeds)),
                'episodes_per_seed': episodes_per_seed,
                'mean_return': float(np.mean(all_returns)),
                'min_return': float(np.min(all_returns)),
                'max_return': float(np.max(all_returns)),
                'mean_action_abs': float(np.mean(all_action_abs)),
                # v0.3新增：x displacement相关统计
                'mean_x_displacement': float(np.mean(all_x_displacements)) if all_x_displacements else None,
                'mean_x_velocity': float(np.mean(all_x_velocities)) if all_x_velocities else None,
                # v0.4新增：wall time统计
                'mean_wall_time_sec': float(np.mean(all_wall_times)) if all_wall_times else None,
                'params_json': json.dumps(params),
                'dominant_failure_hint': max(set(all_failure_hints), key=all_failure_hints.count)
            }
            sweep_results.append(candidate_result)

            print(f"   📊 Candidate {candidate_id}: mean_return={candidate_result['mean_return']:.2f}")

    return sweep_results


def save_sweep_results(sweep_results, output_path='runs/sweep_results.csv'):
    """
    Save sweep results to CSV.

    Args:
        sweep_results: List of candidate results
        output_path: Output CSV path
    """
    if not sweep_results:
        print("❌ No sweep results to save")
        return

    # Ensure runs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame and sort by mean_return (descending)
    df = pd.DataFrame(sweep_results)
    df_sorted = df.sort_values('mean_return', ascending=False)

    # Save to CSV
    df_sorted.to_csv(output_path, index=False)
    print(f"\n✅ Sweep results saved to: {output_path}")

    # Print top candidates
    print(f"\n🏆 Top 3 candidates:")
    for idx, row in df_sorted.head(3).iterrows():
        print(f"   {idx + 1}. {row['candidate_id']}: mean_return={row['mean_return']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Run CPG parameter sweep')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5', help='Environment ID')
    parser.add_argument('--preset', type=str, choices=['tiny', 'small', 'gait_tiny', 'alternating_local', 'phase_refine', 'phase_top_regression', 'base_finetune', 'mpc_tiny'], default='tiny',
                       help='Parameter preset')
    parser.add_argument('--policy', type=str, choices=['cpg_pd', 'mpc'], default='cpg_pd',
                       help='Policy type')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Episodes per seed')
    parser.add_argument('--seeds', type=str, default='0',
                       help='Seeds (comma-separated)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Max steps per episode')

    args = parser.parse_args()

    # Run sweep
    sweep_results = run_param_sweep(
        env_id=args.env,
        preset_name=args.preset,
        episodes_per_seed=args.episodes,
        seeds=args.seeds,
        max_steps=args.max_steps,
        policy_type=args.policy
    )

    # Save sweep results
    if sweep_results:
        save_sweep_results(sweep_results)
    else:
        print("❌ No sweep results generated")


if __name__ == '__main__':
    main()