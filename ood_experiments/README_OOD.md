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

## 依赖

```
pip install numpy scipy torch
```
无CPLEX环境下流体LP自动切换到scipy/HiGHS后端（`Class_FAJSP/fluid_lp.py`，模型与docplex等价）。

## 运行步骤

```bash
# 1. 生成算例 + 求流体下界 + 跑5条PDR取最优 (几分钟)
python ood_experiments/run_lb_pdr.py
#    → results/lb_pdr.csv, 算例数据写入 data/D*/

# 2. D1-D12轮询训练 (每实例独立agent, 逐episode轮流; 与agent/PPO.py的轮训方式一致)
python ood_experiments/run_training.py --episodes 1000
#    → results/train_curves.csv (逐episode追加), results/ppo_ood_D*.pth
#    创建 ood_experiments/STOP.txt 可随时安全终止
#    可用 --instances D1,D2,D3 只训练部分算例

# 3. 汇总: FNIRL最终结果取最后收敛窗口(默认最后20% episode)的最小值
python ood_experiments/collect_results.py
#    → results/final_table.csv / final_table.md
#    (含 vs bestPDR 改进% 和 vs 流体下界 gap%)
```

## 计算量提示

单episode耗时与总工件数近似成正比（4核CPU上D1约21s/episode）。
D4-D6因消耗系数在装配链上乘性传播，总工件数达1212/2424/3636，
建议在GPU机器上训练，或用 `--instances` 分组并行。

## 论文对接

`final_table.csv` 中各列对应论文Table "Generalization under distributional shifts"：
- `fluid_lb` → Fluid LB列
- `fnirl_final` → FNIRL列（最后收敛窗口最小值；`window_mean/std`反映收敛稳定性）
- `best_pdr`/`best_rule` → Best PDR列
- `impr_vs_pdr_pct` / `gap_to_lb_pct` → 正文引用的改进与gap百分比
DRLG基线如需对比，可按同样方式将agent/A3C.py改造为无头版本训练。
