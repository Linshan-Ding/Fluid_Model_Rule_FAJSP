# OOD泛化实验 (论文修订: D1-D12分布偏移算例)

对应论文R1修订版第6.10节 "Generalization under Distributional Shifts"。

## 算例设计

全部基于C13-C15的中规模配置 (M_p=8, M_a=4, P=4, R=8, J_r=4)，每组只改动一个生成因子：

| 组 | 算例 | 偏移 | 批量N_p |
|---|---|---|---|
| Shift-PT | D1-D3 | 加工时间 U[10,50]（训练分布U[10,20]） | 2/4/6 |
| Shift-alpha | D4-D6 | 装配消耗系数 U[1,5]（训练分布U[1,2]） | 2/4/6 |
| Shift-BOM | D7-D9 | 三层BOM：组件→2个子装配体→成品 | 2/4/6 |
| Shift-Elig | D10-D12 | 机器可选：sparse每工序1-2台 / dense全部 | 4/6/4 |

实现方式：`Class_FAJSP/Instance_generate.py` 新增分布参数
（`pt_range`/`alpha_range`/`elig_mode`/`bom_mode`/`rng_seed`/`instance_name`），
默认值与原实现完全一致，C1-C27不受影响。
算例配置定义在 `configs.py`，数据文件在 `data/D*/`。

## 训练方式：顺序轮询 + 共享网络

单一共享的策略/价值网络（结构与超参数与`agent/PPO.py`一致），
全局共训练 **1200个episode**，每个episode按 D1→D2→...→D12→D1→... 的顺序
切换到下一个算例（每个算例各分得100个episode）。
状态矩阵统一padding到88×88、动作特征维度一致（178维），
因此单一网络可跨算例共享参数——这也是原仓库统一状态矩阵维度的设计意图。

最终结果 = 每个算例**最后阶段收敛窗口**（默认最后20%的episode，至少30个）
内的最小完工时间，同时报告窗口均值/标准差以反映收敛稳定性。

## 依赖

```bash
pip install numpy scipy torch
```
无CPLEX环境下流体LP自动切换到scipy/HiGHS后端（`Class_FAJSP/fluid_lp.py`，模型与docplex等价）。

## 运行步骤

```bash
# 1. 生成算例 + 求流体下界 + 跑5条PDR取最优 (几分钟)
python ood_experiments/run_lb_pdr.py
#    → results/lb_pdr.csv (本仓库已附带实测结果), 算例数据写入 data/D*/

# 2. 顺序轮询训练 (共1200个episode, 每episode切换下一个算例)
python ood_experiments/run_training.py --total-episodes 1200
#    → results/train_curves.csv   逐episode追加(global_episode/instance/episode/makespan/...)
#    → results/ppo_ood_shared.pth 共享模型断点(默认每12个episode保存)
#    中断后可断点续训:  python ood_experiments/run_training.py --resume
#    随时安全终止:      创建 ood_experiments/STOP.txt
#    只训练部分算例:    --instances D1,D2,D3

# 3. 汇总: 收敛窗口统计 + vs最优PDR改进% + vs流体下界gap%
python ood_experiments/collect_results.py
#    → results/final_table.csv / final_table.md
```

## 计算量提示

单episode耗时与算例总工件数近似成正比（4核CPU上D1约16-20s/episode）。
D4-D6因消耗系数在装配链上乘性传播，总工件数达1212/2424/3636，
是训练耗时的大头，建议在GPU/多核机器上运行完整的1200个episode。

## 结果与论文表格的对应关系

`results/final_table.csv` 各列对应论文表 "Generalization under distributional shifts"：

| CSV列 | 论文表列 |
|---|---|
| `fluid_lb` | Fluid LB（流体下界，逐算例重解LP得到） |
| `fnirl_final` | FNIRL（收敛窗口最小完工时间） |
| `window_mean` / `window_std` | 收敛稳定性（正文引用） |
| `best_pdr` / `best_rule` | Best PDR（5条规则逐算例取最优） |
| `impr_vs_pdr_pct` | 正文引用的对最优PDR改进百分比 |
| `gap_to_lb_pct` | 正文引用的对流体下界gap百分比 |

DRLG基线如需对比，可按`ppo_headless.py`的模式将`agent/A3C.py`改造为无头版本，
用相同的顺序轮询协议训练。
