# 快速启动指南 (v0.1.1)

## 🚀 5 分钟快速开始

### Python 版本要求

- **推荐**: Python 3.10 或 3.11
- **兼容**: Python 3.12
- **不推荐**: Python 3.14+（MuJoCo/Gymnasium 兼容性问题）

### 第一步：安装依赖（2 分钟）

```bash
# 进入项目目录
cd genesis_robot_hl_repro

# 安装依赖
pip install -r requirements.txt

# 验证安装
python3 -c "import gymnasium; print('✅ Gymnasium installed')"
python3 -c "import mujoco; print('✅ MuJoCo installed')"
```

**v0.1.1 新增**: 改进的可复现性（action_space.seed）和诊断能力（action 统计）

### 第二步：运行 Baseline（1 分钟）

```bash
# Random 策略基准测试
python3 run_eval.py --policy random --episodes 3 --trial-name baseline_random
```

### 第三步：运行 CPG/PD 策略（1 分钟）

```bash
# 周期步态策略
python3 run_eval.py --policy cpg_pd --episodes 5 --trial-name cpg_v01
```

### 第四步：查看结果（1 分钟）

```bash
# 查看日志
cat runs/trials.jsonl | jq

# 查看汇总
cat runs/summary.csv

# 生成图表
python3 tools/plot_summary.py
open runs/summary_curve.png
```

## 🔧 常见问题解决

### Q: "gymnasium not found"
```bash
pip install gymnasium[mujoco]
```

### Q: "mujoco license error"
```bash
# MuJoCo 现在是开源的，但可能需要：
pip install mujoco
# 或使用免费的个人许可证
```

### Q: "HalfCheetah-v5 not found"
```bash
# 系统会自动 fallback 到 v4
# 如果都不可用，检查 gymnasium 版本
pip show gymnasium
```

## 📊 预期输出

### 成功运行的标志：
```
🏃 Running trial: cpg_v01
   Environment: HalfCheetah-v5 -> HalfCheetah-v4
   Policy: cpg_pd
   Episodes: 5
   Seed: 0

📊 Results:
   Mean Return: -123.45
   Min/Max Return: -150.00 / -100.00
   Mean Steps: 200.3

💡 Failure Hints:
   Episode 0: moving_backward_or_high_cost (return=-123.4)
   Episode 1: moving_backward_or_high_cost (return=-145.6)
   ...

✅ Results saved:
   - runs/trials.jsonl
   - runs/summary.csv
```

## 🎯 参数调优示例

### 调整步态频率：
```bash
python3 run_eval.py --policy cpg_pd --params-json '{"phase_speed": 0.15}'
```

### 调整振幅：
```bash
python3 run_eval.py --policy cpg_pd --params-json '{"hip_amp": 0.4, "knee_amp": 0.3}'
```

### 组合调优：
```bash
python3 run_eval.py --policy cpg_pd --params-json '{
  "phase_speed": 0.2,
  "hip_amp": 0.5,
  "knee_amp": 0.3,
  "ankle_amp": 0.15
}'
```

## 🔍 下一步实验

1. **多 seed 稳定性测试**:
   ```bash
   for seed in 0 1 2 3 4; do
     python3 run_eval.py --policy cpg_pd --seed $seed --trial-name cpg_seed_$seed
   done
   ```

2. **参数扫描**:
   ```bash
   for speed in 0.05 0.1 0.15 0.2; do
     python3 run_eval.py --policy cpg_pd --params-json "{\"phase_speed\": $speed}" --trial-name speed_$speed
   done
   ```

3. **性能曲线**:
   ```bash
   python3 tools/plot_summary.py
   open runs/summary_curve.png
   ```

---

**祝实验顺利！🎉**

如有问题，请查看 `README.md` 或 `PROJECT_SUMMARY.md`
