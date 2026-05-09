# Genesis Robot HL Repro v0.4.6

**A small auditable runtime-learning demo for MuJoCo HalfCheetah**

This project demonstrates how a heuristic robot controller can evolve through feedback logs, gait search, regression checks, and a lightweight residual MPC layer.

这是一个小型可审计智能运行层 demo。它不追求 SOTA 分数，而是展示一个启发式机器人控制器如何通过真实反馈、日志、诊断、步态搜索、回归验证和轻量 residual MPC 逐步形成可解释改进。

---

## What This Demo Shows

- **Random baseline** - Uniform random action sampling
- **CPG/PD heuristic base gait** - Central Pattern Generator + Proportional-Derivative control
- **Parameter and gait search infrastructure** - Systematic exploration of control parameters
- **Stable positive-return base gait** - Consistent forward motion with CPG/PD
- **Lightweight residual MPC layer** - Model Predictive Control on top of stable CPG base
- **Clean CPG vs MPC comparison** - Automated comparison tools for performance evaluation
- **Auditable logs** - trials.jsonl, summary.csv, comparison_results.csv

---

## Current v0.4.6 Demo Result

**Clean CPG vs MPC comparison on HalfCheetah-v5:**

### 100-step episodes
| Policy  | Return | X-Displacement | Velocity  |
|---------|--------|----------------|-----------|
| CPG ft015 | 3.53   | 0.231          | 0.00231   |
| MPC mpc_004 | 7.06   | 0.405          | 0.00405   |

### 300-step episodes
| Policy  | Return | X-Displacement | Velocity  |
|---------|--------|----------------|-----------|
| CPG ft015 | 11.00  | 0.716          | 0.00239   |
| MPC mpc_004 | 15.84  | 0.950          | 0.00317   |

**Interpretation:**
The lightweight residual MPC layer improves short/mid-horizon return and displacement over the pure CPG base gait, at significantly higher compute cost.

---

## Install

```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies
# - gymnasium[mujoco]
# - numpy
# - matplotlib
# - pandas
```

---

## Run Clean Comparison

```bash
# Clean previous results
python tools/clean_runs.py

# Run CPG vs MPC comparison (100 and 300 steps)
python tools/compare_cpg_mpc.py --steps 100,300 --seed 0 --clean

# Results will be saved to runs/comparison_results.csv
```

---

## Run Individual Policies

```bash
# Random baseline (1 episode)
python run_eval.py --policy random --episodes 1 --seed 0

# CPG/PD base gait with default parameters
python run_eval.py --policy cpg_pd --episodes 1 --seed 0

# CPG/PD with custom parameters
python run_eval.py --policy cpg_pd --episodes 1 --seed 0 --cpg-freq 1.5 --cpg-amp 0.2

# Lightweight residual MPC (100 steps)
python run_eval.py --policy mpc --episodes 1 --seed 0 --max-steps 100

# MPC with custom parameters
python run_eval.py --policy mpc --episodes 1 --seed 0 --max-steps 200 \
  --cpg-freq 1.5 --cpg-amp 0.2 --mpc-horizon 3 --mpc-candidates 12
```

---

## Project Status

**This is not a SOTA robotics result.**
It is a small auditable runtime-learning demo.

### Design Principles
- **CPU-only** - No neural networks, no GPU required
- **Transparent** - Every component is interpretable
- **Lightweight** - Short horizons, small candidate sets
- **Reproducible** - Fixed seeds, deterministic policies
- **Auditable** - Complete logs and diagnostics

### Technical Approach
- **Base Layer**: CPG/PD gait generator (stable, rhythmic motion)
- **Improvement Layer**: Residual MPC (short-horizon local optimization)
- **Evaluation**: Systematic comparison with failure mode classification

---

## File Structure

```
genesis_robot_hl_repro/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── run_eval.py           # Main evaluation script
├── policies/             # Policy implementations
│   ├── __init__.py
│   ├── random_policy.py           # Random baseline
│   ├── halfcheetah_cpg_pd.py      # CPG/PD base gait
│   └── halfcheetah_residual_mpc.py # Residual MPC policy
├── tools/                # Analysis and utilities
│   ├── __init__.py
│   ├── clean_runs.py              # Clean experiment results
│   ├── compare_cpg_mpc.py         # CPG vs MPC comparison
│   ├── param_sweep.py             # Parameter sweep tool
│   ├── plot_summary.py            # Plot summary statistics
│   └── plot_sweep.py              # Plot sweep results
└── runs/                 # Experiment results (gitignored except .gitkeep)
    └── .gitkeep
```

---

## Key Parameters

### CPG/PD Base Gait
- `--cpg-freq`: CPG frequency (Hz), default 1.0
- `--cpg-amp`: CPG amplitude, default 0.15
- `--pd-kp`: PD proportional gain, default 20.0
- `--pd-kd`: PD derivative gain, default 0.5

### Residual MPC
- `--mpc-horizon`: Optimization horizon (steps), default 3
- `--mpc-candidates`: Number of candidate sequences, default 12
- `--mpc-action-scale`: Residual action scale, default 0.1

---

## Evaluation Results Format

All results are saved in the `runs/` directory:

- `trials.jsonl` - Individual trial results (one JSON object per line)
- `summary.csv` - Aggregated summary statistics
- `comparison_results.csv` - CPG vs MPC comparison results
- `sweep_results.csv` - Parameter sweep results

Each trial record includes:
- `trial_id`: Unique identifier
- `policy_name`: Policy used (random/cpg_pd/mpc)
- `episode_return`: Total cumulative reward
- `steps`: Number of steps taken
- `x_displacement`: Final forward displacement
- `mean_x_velocity`: Average forward velocity
- `failure_hint`: Diagnostic failure mode classification
- `parameters`: Policy parameters used

---

## Hardware Requirements

- **CPU**: Any modern multi-core processor
- **Memory**: < 1GB RAM
- **Storage**: < 100MB
- **GPU**: Not required (CPU-only design)

---

## Version History

- **v0.4.6** (2026-05-09) - Comparison tool hotfix, failure hint improvements
- **v0.4.5** (2026-05-09) - Added comparison tools and run cleanup utilities
- **v0.4.4** (2026-05-09) - Fixed wrapper pollution in MPC internal rollouts
- **v0.4.3** (2026-05-09) - Fixed float phase handling bug
- **v0.4.2** (2026-05-09) - Fixed phase pollution in MPC base action
- **v0.4.1** (2026-05-09) - Fixed MPC policy constructor and environment binding
- **v0.4.0** (2026-05-09) - Initial release with CPG/PD base gait and residual MPC

---

## License

MIT License - See LICENSE file for details

---

## Contact & Contributions

This is a research demonstration project. For questions or improvements, please open an issue or submit a pull request.
