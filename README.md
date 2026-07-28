# Fluid_Model_Rule_FAJSP

面向高品种小批量制造的柔性装配作业车间调度（FAJSP-HMLV）：
流体网络模型 + 优先调度规则 + 深度强化学习（FNIRL相关实验代码）。

配套论文: *Fluid Network-Informed Reinforcement Learning for Flexible Assembly
Job Shop Scheduling in High-Mix Low-Volume Manufacturing* (修订版R1)。

## 目录结构

```
Class_FAJSP/
  Instance_generate.py   实例生成器（支持分布参数覆盖, 用于OOD实验）
  class_fajsp.py         FAJSP类: 工件/机器对象、流体网络LP模型
  fluid_lp.py            流体LP的scipy/HiGHS后端（无CPLEX时自动启用, 与docplex等价）
  assemble_method.py     BOM结构与装配序列生成工具
Class_AJSP/              装配作业车间(AJSP)相关类
env/
  env.py                 MDP仿真环境（多视图状态88x88、候选动作、流体归一化奖励）
agent/
  PPO.py                 PPO训练主程序（含visdom可视化, 本地开发用）
  A2C.py A3C.py AC.py DQN.py DDQN.py   对比RL算法
  CNN.py ATN.py MLP.py   策略网络组件（CNN状态编码/自注意力动作交互/MLP打分）
  PPO_eval.py            已训练模型评估
schedule_rule/
  schedule_rule.py       优先调度规则（SPT/FIFO/MRT/MRT_SPT/MRT_FIFO等）
ood_experiments/         OOD分布偏移泛化实验（论文6.10节, 见README_OOD.md）
  pretrained.py            预训练checkpoint加载(按state_dict形状动态重建网络)
  run_eval_pretrained.py   9个预训练策略求解D1-D12, best-of-9作为FNIRL结果
  run_lb_pdr.py            流体下界 + 5条PDR取最优
  run_training.py          备选: D组从头顺序轮询训练
models/
  ppo_fajsp_model_8_4_*.pth  论文C10-C18九个中等规模算例训练完毕的策略网络参数
data/
  4_2_* 8_4_* 12_6_*     C1-C27基准算例（命名: Mp_Ma_P_R_Jr_Np）
  D1_* ... D12_*         D1-D12分布偏移算例
```

## 环境依赖

```bash
pip install numpy scipy torch
# 可选: pip install docplex cplex   (有CPLEX时流体LP使用docplex, 否则自动用scipy/HiGHS)
# 可选: pip install visdom matplotlib (仅agent/PPO.py本地可视化训练需要)
```

## 快速开始

### 1. C1-C27基准实验（原有流程）

```bash
python Class_FAJSP/Instance_generate.py     # 生成C1-C27算例数据
python schedule_rule/schedule_rule.py       # 调度规则基线
python agent/PPO.py                         # PPO训练（需先启动 python -m visdom.server）
```

注意: 跨算例轮询训练前确保状态矩阵维度统一（当前统一padding到88x88）。

### 2. D1-D12 OOD泛化实验（论文修订新增）

评测协议: 加载`models/`下9个预训练策略(C10-C18训练所得)直接求解每个D算例,
**零重训**, 每个算例取9个结果中的最优值作为FNIRL结果。

```bash
# 流体下界 + 5条PDR取最优（结果已随仓库提供: ood_experiments/results/lb_pdr.csv）
python ood_experiments/run_lb_pdr.py

# 预训练策略best-of-9评测, 直接生成论文表格数据
python ood_experiments/run_eval_pretrained.py
#   → ood_experiments/results/final_table_pretrained.csv
```

备选(从头训练): `python ood_experiments/run_training.py --total-episodes 1200`

详细说明（算例设计、评测协议、结果与论文表格的列对应）见
[`ood_experiments/README_OOD.md`](ood_experiments/README_OOD.md)。
