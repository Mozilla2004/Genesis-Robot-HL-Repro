# Genesis Robot HL Repro v0.1.1

**HalfCheetah CPU-only Heuristic Learning Prototype**

## 🎯 目标

这是一个 **CPU-only** 的 HalfCheetah Heuristic Learning 原型，**不训练神经网络**，**不使用 GPU**。

核心目标不是追求最高分，而是复现智能运行层的闭环：

```
程序策略 → 环境运行 → 日志记录 → 失败模式观察 → 参数/策略调整 → 多 seed 回归测试 → summary 输出
```

## 📦 包含内容

- ✅ **Random Baseline**: 随机策略基准
- ✅ **CPG/PD Heuristic**: 周期步态生成器 + 姿态反馈（纯 numpy）
- ✅ **试验记录**: trials.jsonl（per-episode 日志）
- ✅ **汇总统计**: summary.csv（per-trial 汇总）
- ✅ **失败分析**: failure_hint（轻量级失败模式识别）
- ✅ **可视化**: summary_curve.png（性能曲线）

## 🚀 快速开始

### Python 版本要求

- **推荐**: Python 3.10 或 3.11
- **兼容**: Python 3.12
- **不推荐**: Python 3.14+（MuJoCo/Gymnasium 兼容性问题）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 需要安装 MuJoCo 物理引擎。如果遇到 MuJoCo 许可问题，请参考 Gymnasium 文档。

### 2. 运行 Random Baseline

```bash
python run_eval.py --policy random --episodes 3 --seed 0 --trial-name smoke_random
```

### 3. 运行 CPG/PD Policy

```bash
python run_eval.py --policy cpg_pd --episodes 5 --seed 0 --trial-name cpg_v01
```

### 4. 自定义 CPG 参数

```bash
python run_eval.py --policy cpg_pd --episodes 5 --params-json '{"phase_speed": 0.15, "hip_amp": 0.4}'
```

### 5. 生成性能曲线

```bash
python -m tools.plot_summary
# 或直接
python tools/plot_summary.py
```

## 📁 项目结构

```
genesis_robot_hl_repro/
├── README.md              # 本文件
├── requirements.txt       # 依赖列表
├── run_eval.py            # 主评估脚本
├── policies/              # 策略实现
│   ├── __init__.py
│   ├── random_policy.py   # 随机策略
│   └── halfcheetah_cpg_pd.py  # CPG/PD 启发式策略
├── tools/                 # 工具脚本
│   ├── __init__.py
│   └── plot_summary.py    # 绘制性能曲线
└── runs/                  # 试验结果目录
    ├── trials.jsonl       # Episode 级别日志
    ├── summary.csv        # Trial 级别汇总
    └── summary_curve.png  # 性能曲线图
```

## 🔧 命令行参数

```bash
python run_eval.py [OPTIONS]

选项:
  --env ENV               环境ID (默认: HalfCheetah-v5)
  --policy {random,cpg_pd} 策略名称 (默认: random)
  --episodes N            Episode 数量 (默认: 3)
  --seed N                随机种子 (默认: 0)
  --trial-name NAME       试验名称 (默认: 自动生成)
  --max-steps N           最大步数 (默认: 环境默认)
  --render-video          渲染视频 (TODO)
  --params-json JSON      CPG 参数 (JSON 字符串)
```

## 🧪 失败模式分析（failure_hint）

系统会自动分析每次 episode 的失败模式：

| Hint | 含义 |
|------|------|
| `early_termination_or_unstable` | 步数很少，可能不稳定 |
| `moving_backward_or_high_cost` | 负回报，可能倒退或代价过高 |
| `survives_but_poor_forward_progress` | 存活但前进效果差 |
| `working_candidate` | 高回报，可能是有效参数 |

## 📊 输出文件格式

### trials.jsonl（每行一个 JSON）

```json
{
  "timestamp": "2026-05-09T...",
  "trial_name": "cpg_v01",
  "env_id": "HalfCheetah-v5",
  "actual_env_id": "HalfCheetah-v4",
  "policy": "cpg_pd",
  "seed": 0,
  "episode": 0,
  "return": -123.45,
  "steps": 123,
  "params": {"phase_speed": 0.1, ...},
  "notes": "",
  "failure_hint": "moving_backward_or_high_cost"
}
```

### summary.csv（每个 trial 一行）

```csv
trial_name,env_id,actual_env_id,policy,seed,episodes,mean_return,min_return,max_return,mean_steps,params_json
cpg_v01,HalfCheetah-v5,HalfCheetah-v4,cpg_pd,0,5,-100.2,-150.0,-50.0,200.3,"{\"phase_speed\":0.1}"
```

## 🎓 CPG/PD 策略说明

**Central Pattern Generator (CPG)**:
- 使用 sin/cos 生成周期步态
- 左右腿相位差 π（交替运动）
- 髋关节/膝关节/踝关节不同振幅

**Proportional-Derivative (PD) Control**:
- 简单的姿态反馈（pitch correction）
- 阻尼项（简化版）

**可调参数**:
- `phase_speed`: 步态频率
- `hip_amp`, `knee_amp`, `ankle_amp`: 各关节振幅
- `action_scale`: 动作缩放
- `pitch_gain`: 姿态反馈增益
- `damping_gain`: 阻尼增益

## 🚧 版本说明

**v0.1.1** (当前版本):
- ✅ **Hotfix**: 修复重复 reset 问题
- ✅ **可复现性**: 添加 action_space.seed()
- ✅ **诊断能力**: 增加 action 统计（abs_mean/min/max）
- ✅ **失败分析**: 改进 failure_hint，增加 action_too_weak
- ✅ **CPG/PD 改进**: 修复 observation mapping，实现真实阻尼计算
- ✅ **文档**: 更新 Python 版本建议

**v0.1** (初始版本):
- ✅ 基础闭环搭建
- ✅ Random baseline
- ✅ CPG/PD 启发式策略
- ✅ 日志记录与汇总
- ✅ 轻量级失败分析

**v0.2** (计划中):
- 参数扫描（grid search）
- 更丰富的失败模式分析
- 多 seed 稳定性评估

**v0.3** (计划中):
- Staged-tree MPC
- 自适应参数调整
- 更高级的启发式规则

## ⚠️ 注意事项

1. **不追求高分**: v0.1.1 重点是真实环境可运行和日志可诊断，不是性能优化
2. **CPU-only**: 所有计算都在 CPU 上完成，无需 GPU
3. **MuJoCo 依赖**: 需要正确安装 MuJoCo 和 Gymnasium
4. **版本兼容**: 自动处理 HalfCheetah-v4/v5 差异
5. **Python 版本**: 推荐 Python 3.10/3.11，避免 3.14+ 兼容性问题

## 🔍 下一步建议

1. **运行 smoke tests**: 验证环境配置正确
2. **调整 CPG 参数**: 尝试不同的 phase_speed, amp 等参数
3. **观察 failure_hint**: 分析失败模式，指导下一轮参数调整
4. **积累试验数据**: 多次运行后观察 summary_curve.png 的趋势

## 📖 参考资料

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MuJoCo Physics](https://mujoco.org/)
- Central Pattern Generation: 生物运动控制理论

---

**作者**: Genesis Robot HL Team
**版本**: v0.1.1
**日期**: 2026-05-09
**协议**: MIT
