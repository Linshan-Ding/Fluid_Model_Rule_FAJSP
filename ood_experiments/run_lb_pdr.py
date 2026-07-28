# -*- coding: utf-8 -*-
"""
对D1-D12算例: 重解流体LP得下界C_max^F, 跑5条PDR取最优。
结果写入 ood_experiments/results/lb_pdr.csv, 并生成算例数据文件到 data/。

用法: python ood_experiments/run_lb_pdr.py
"""
import os
import sys
import csv
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.env import FAJSP_Environment  # noqa: E402
from ood_experiments.configs import OOD_CONFIGS, ORDER  # noqa: E402
from ood_experiments.pdr import PDR_RULES  # noqa: E402


def main():
    out_dir = os.path.join(REPO_ROOT, 'ood_experiments', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'lb_pdr.csv')

    rows = []
    for name in ORDER:
        cfg = OOD_CONFIGS[name]
        env = FAJSP_Environment(**cfg['env_kwargs'])
        env.reset()
        fluid_lb = env.fluid_completed_time
        total_units = sum(len(env.part_dict[p].part_unprocessed_list) for p in env.part_dict)
        op_types = len(env.kind_task_tuple)

        # 写出算例数据文件(可复现)
        try:
            os.makedirs(os.path.join(REPO_ROOT, 'data'), exist_ok=True)
            cwd = os.getcwd()
            os.chdir(os.path.join(REPO_ROOT, 'ood_experiments'))
            env.write_file()  # 写入 ../data/<instance_name>/
            os.chdir(cwd)
        except Exception as e:  # 数据写出失败不影响评测
            print(f"[warn] {name} 数据文件写出失败: {e}")

        pdr_results = {}
        for rule_name, rule_fn in PDR_RULES.items():
            t0 = time.time()
            ct, adt = rule_fn(env)
            pdr_results[rule_name] = ct
            print(f"{name:<4} {rule_name:<9} makespan={ct:<8.1f} 用时{time.time()-t0:5.1f}s 平均决策{adt*1000:6.2f}ms")

        best_rule = min(pdr_results, key=pdr_results.get)
        best_pdr = pdr_results[best_rule]
        row = dict(instance=name, shift=cfg['shift'],
                   op_types=op_types, total_units=total_units,
                   fluid_lb=round(fluid_lb, 1),
                   **{f'pdr_{k}': v for k, v in pdr_results.items()},
                   best_pdr=best_pdr, best_rule=best_rule)
        rows.append(row)
        print(f"{name:<4} LB={fluid_lb:8.1f}  bestPDR={best_pdr:.0f} ({best_rule})  "
              f"工序类型{op_types} 总工件数{total_units}")

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已写入 {out_csv}")


if __name__ == '__main__':
    main()
