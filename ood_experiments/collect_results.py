# -*- coding: utf-8 -*-
"""
汇总OOD实验结果: 合并流体下界/PDR结果与训练曲线, 生成论文表格数据。

FNIRL最终结果 = 训练最后阶段收敛窗口(默认最后20%的episode, 至少30个)内的
最小完工时间(同时报告窗口均值/标准差以反映收敛稳定性)。

用法: python ood_experiments/collect_results.py [--window-frac 0.2]
输出: results/final_table.csv 与 results/final_table.md
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ood_experiments.configs import ORDER  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, 'ood_experiments', 'results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-frac', type=float, default=0.2,
                        help='收敛窗口占训练episode总数的比例')
    parser.add_argument('--window-min', type=int, default=30,
                        help='收敛窗口最少episode数')
    args = parser.parse_args()

    # 读取LB与PDR结果
    lb_pdr = {}
    with open(os.path.join(RESULT_DIR, 'lb_pdr.csv')) as f:
        for row in csv.DictReader(f):
            lb_pdr[row['instance']] = row

    # 读取训练曲线
    curves = defaultdict(list)
    curves_path = os.path.join(RESULT_DIR, 'train_curves.csv')
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            for row in csv.DictReader(f):
                curves[row['instance']].append((int(row['episode']), float(row['makespan'])))

    rows = []
    for name in ORDER:
        if name not in lb_pdr:
            continue
        info = lb_pdr[name]
        lb = float(info['fluid_lb'])
        best_pdr = float(info['best_pdr'])

        entry = dict(instance=name, shift=info['shift'], fluid_lb=lb,
                     best_pdr=best_pdr, best_rule=info['best_rule'])
        series = sorted(curves.get(name, []))
        if series:
            cts = [ct for _, ct in series]
            n = len(cts)
            w = max(args.window_min, int(n * args.window_frac))
            window = cts[-min(w, n):]
            fnirl = min(window)
            entry.update(
                episodes=n,
                window_size=len(window),
                fnirl_final=fnirl,
                window_mean=round(statistics.mean(window), 1),
                window_std=round(statistics.pstdev(window), 1),
                global_best=min(cts),
                impr_vs_pdr_pct=round((best_pdr - fnirl) / best_pdr * 100, 1),
                gap_to_lb_pct=round((fnirl - lb) / lb * 100, 1),
            )
        rows.append(entry)

    if not rows:
        print("无结果可汇总")
        return

    fieldnames = list(rows[-1].keys())
    out_csv = os.path.join(RESULT_DIR, 'final_table.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # markdown表格
    out_md = os.path.join(RESULT_DIR, 'final_table.md')
    with open(out_md, 'w') as f:
        f.write('| ' + ' | '.join(fieldnames) + ' |\n')
        f.write('|' + '---|' * len(fieldnames) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(r.get(k, '')) for k in fieldnames) + ' |\n')

    print(f"已写入 {out_csv} 和 {out_md}\n")
    with open(out_md) as f:
        print(f.read())


if __name__ == '__main__':
    main()
