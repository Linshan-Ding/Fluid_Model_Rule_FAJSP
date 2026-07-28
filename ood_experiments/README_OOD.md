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

## 评测协议（主流程）：预训练策略best-of-9，零重训

`models/` 下是论文中在C10-C18九个中等规模算例上训练完毕的9组策略网络参数
（结构为论文配置：CNN卷积核[64,32]、MLP隐层[64,32]、自注意力178维）。
OOD评测**不做任何重训或微调**：将9个预训练策略逐一用于每个D算例的贪婪调度
（每个算例仅按标准推理流程重解一次流体LP），每个算例得到9个makespan，
**取最优值作为FNIRL的结果**。

```bash
pip install numpy scipy torch

# 1. 流体下界 + 5条PDR取最优（结果已随仓库附带: results/lb_pdr.csv）
python ood_experiments/run_lb_pdr.py

# 2. 加载9个预训练模型求解D1-D12, best-of-9作为FNIRL结果
python ood_experiments/run_eval_pretrained.py
#    → results/eval_pretrained.csv          每个(算例,模型)的makespan与决策时间
#    → results/final_table_pretrained.csv   论文表格数据(见下方列对应)
```

`pretrained.py` 按checkpoint的state_dict形状动态重建网络结构，
不依赖 `agent/CNN.py` 等文件的当前实现（当前为精简版，与训练时结构不同）。

无CPLEX环境下流体LP自动切换到scipy/HiGHS后端
（`Class_FAJSP/fluid_lp.py`，模型与docplex等价）。

## 结果与论文表格的对应关系

`results/final_table_pretrained.csv` 各列对应论文表
"Generalization under distributional shifts"：

| CSV列 | 论文表列 |
|---|---|
| `fluid_lb` | Fluid LB（流体下界，逐算例重解LP得到） |
| `fnirl_best_of_9` | FNIRL（9个预训练策略的最优makespan） |
| `best_model` | 取得最优值的预训练模型（附录/正文可引用） |
| `avg_decision_ms` | 平均每步决策时间 |
| `best_pdr` / `best_rule` | Best PDR（5条规则逐算例取最优） |
| `impr_vs_pdr_pct` | 正文引用的对最优PDR改进百分比 |
| `gap_to_lb_pct` | 正文引用的对流体下界gap百分比 |

## 备选流程：顺序轮询训练（从头训练）

如需在D组上从头训练（而非零重训迁移评测），保留了顺序轮询训练流程：
单一共享策略网络，全局1200个episode，每个episode按D1→...→D12顺序切换算例。

```bash
python ood_experiments/run_training.py --total-episodes 1200   # 断点续训: --resume
python ood_experiments/collect_results.py                      # 收敛窗口统计
```

注意：`run_training.py` 使用 `agent/` 目录当前的精简网络组件，
与 `models/` 下checkpoint的结构不同，二者的模型文件不可互换。

## 计算量提示

贪婪评测的耗时与算例总工件数成正比。D4-D6因消耗系数在装配链上乘性传播，
总工件数达1212/2424/3636，占评测总耗时的大头（完整best-of-9评测约1-1.5小时，4核CPU）。
