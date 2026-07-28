# -*- coding: utf-8 -*-
"""
用models/下9个预训练策略(论文C10-C18中等规模算例训练所得)直接求解D1-D12,
每个算例得到9个makespan, 取最优值作为FNIRL的结果(不做任何重训/微调,
仅按标准推理流程为每个算例重解流体LP)。

输出:
  results/eval_pretrained.csv        每个(算例, 模型)的makespan与平均决策时间
  results/final_table_pretrained.csv 论文表格数据: LB / FNIRL(best-of-9) / bestPDR / 改进% / gap%
  results/final_table_pretrained.md  同上(markdown预览)

用法: python ood_experiments/run_eval_pretrained.py [--models-dir models]
"""
import argparse
import csv
import glob
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.env import FAJSP_Environment  # noqa: E402
from ood_experiments.configs import OOD_CONFIGS, ORDER  # noqa: E402
from ood_experiments.pretrained import load_policy, greedy_rollout  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, 'ood_experiments', 'results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=str, default=os.path.join(REPO_ROOT, 'models'))
    parser.add_argument('--instances', type=str, default=','.join(ORDER))
    args = parser.parse_args()

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, '*.pth')))
    if not model_paths:
        print(f"未在 {args.models_dir} 找到模型文件")
        return
    print(f"加载 {len(model_paths)} 个预训练模型:")
    policies = []
    for p in model_paths:
        policy, meta = load_policy(p)
        tag = os.path.basename(p).replace('ppo_fajsp_model_', '').replace('.pth', '')
        policies.append((tag, policy))
        print(f"  {tag:<16} 训练算例config={meta['config']} "
              f"训练最优完工时间={meta['best_completion_time']}")

    names = [n.strip() for n in args.instances.split(',') if n.strip()]
    os.makedirs(RESULT_DIR, exist_ok=True)
    eval_csv = os.path.join(RESULT_DIR, 'eval_pretrained.csv')

    per_instance = {}
    with open(eval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance', 'model', 'makespan', 'avg_decision_ms', 'rollout_seconds'])
        for name in names:
            env = FAJSP_Environment(**OOD_CONFIGS[name]['env_kwargs'])
            results = {}
            for tag, policy in policies:
                t0 = time.time()
                ct, adt = greedy_rollout(policy, env)
                results[tag] = (ct, adt)
                writer.writerow([name, tag, round(ct, 1), round(adt * 1000, 2),
                                 round(time.time() - t0, 1)])
                f.flush()
                print(f"{name:<4} 模型{tag:<16} makespan={ct:<8.0f} "
                      f"平均决策{adt*1000:6.2f}ms 用时{time.time()-t0:5.1f}s", flush=True)
            best_tag = min(results, key=lambda t: results[t][0])
            per_instance[name] = dict(
                fnirl=results[best_tag][0], best_model=best_tag,
                adt_ms=results[best_tag][1] * 1000,
                all9=[results[t][0] for t, _ in policies])
            print(f"{name:<4} ==> best-of-9: {results[best_tag][0]:.0f} (模型{best_tag})", flush=True)

    # 合并LB与PDR结果生成最终表
    lb_pdr = {}
    lb_pdr_csv = os.path.join(RESULT_DIR, 'lb_pdr.csv')
    if os.path.exists(lb_pdr_csv):
        with open(lb_pdr_csv) as f:
            for row in csv.DictReader(f):
                lb_pdr[row['instance']] = row

    fieldnames = ['instance', 'shift', 'fluid_lb', 'fnirl_best_of_9', 'best_model',
                  'avg_decision_ms', 'best_pdr', 'best_rule',
                  'impr_vs_pdr_pct', 'gap_to_lb_pct']
    rows = []
    for name in names:
        if name not in per_instance:
            continue
        r = dict(instance=name, shift=OOD_CONFIGS[name]['shift'],
                 fnirl_best_of_9=round(per_instance[name]['fnirl'], 1),
                 best_model=per_instance[name]['best_model'],
                 avg_decision_ms=round(per_instance[name]['adt_ms'], 2))
        if name in lb_pdr:
            lb = float(lb_pdr[name]['fluid_lb'])
            best_pdr = float(lb_pdr[name]['best_pdr'])
            fnirl = per_instance[name]['fnirl']
            r.update(fluid_lb=lb, best_pdr=best_pdr, best_rule=lb_pdr[name]['best_rule'],
                     impr_vs_pdr_pct=round((best_pdr - fnirl) / best_pdr * 100, 1),
                     gap_to_lb_pct=round((fnirl - lb) / lb * 100, 1))
        rows.append(r)

    out_csv = os.path.join(RESULT_DIR, 'final_table_pretrained.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_md = os.path.join(RESULT_DIR, 'final_table_pretrained.md')
    with open(out_md, 'w') as f:
        f.write('| ' + ' | '.join(fieldnames) + ' |\n')
        f.write('|' + '---|' * len(fieldnames) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(r.get(k, '')) for k in fieldnames) + ' |\n')

    print(f"\n最终表已写入 {out_csv}")
    with open(out_md) as f:
        print(f.read())


if __name__ == '__main__':
    main()
