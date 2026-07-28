# -*- coding: utf-8 -*-
"""
D1-D12轮询训练主程序(云端无头运行)。

轮询方式与agent/PPO.py的train_round_robin一致: 每个算例独立agent,
每轮每个算例各训练1个episode后切换到下一个算例。
训练曲线逐episode追加写入 results/train_curves.csv, 模型定期保存,
最终结果由collect_results.py基于最后阶段收敛窗口统计。

用法:
  python ood_experiments/run_training.py --episodes 300
  可随时创建 ood_experiments/STOP.txt 提前安全终止(结果与模型均已落盘)。
"""
import argparse
import csv
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ood_experiments.configs import OOD_CONFIGS, ORDER  # noqa: E402
from ood_experiments.ppo_headless import InstanceTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300,
                        help='每个算例的训练episode数')
    parser.add_argument('--instances', type=str, default=','.join(ORDER),
                        help='要训练的算例, 逗号分隔, 默认D1-D12全部')
    parser.add_argument('--save-every', type=int, default=20,
                        help='每多少轮保存一次模型')
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    out_dir = os.path.join(REPO_ROOT, 'ood_experiments', 'results')
    os.makedirs(out_dir, exist_ok=True)
    curves_csv = os.path.join(out_dir, 'train_curves.csv')
    stop_file = os.path.join(REPO_ROOT, 'ood_experiments', 'STOP.txt')

    names = [n.strip() for n in args.instances.split(',') if n.strip()]
    print(f"构建训练器: {names}")
    trainers = []
    for name in names:
        t0 = time.time()
        trainer = InstanceTrainer(name, OOD_CONFIGS[name]['env_kwargs'])
        trainers.append(trainer)
        print(f"  {name} 就绪 ({time.time()-t0:.1f}s), "
              f"工序类型{len(trainer.env.kind_task_tuple)}, "
              f"流体下界{trainer.env.fluid_completed_time:.1f}")

    new_file = not os.path.exists(curves_csv)
    fcsv = open(curves_csv, 'a', newline='')
    writer = csv.writer(fcsv)
    if new_file:
        writer.writerow(['instance', 'episode', 'makespan', 'reward', 'seconds'])

    start = time.time()
    round_num = 0
    while True:
        if os.path.exists(stop_file):
            print("\n检测到STOP.txt, 安全终止")
            break
        if all(t.episode_count >= args.episodes for t in trainers):
            break
        for trainer in trainers:
            if trainer.episode_count >= args.episodes:
                continue
            t0 = time.time()
            total_reward, ct, _, _ = trainer.train_one_episode()
            writer.writerow([trainer.name, trainer.episode_count - 1,
                             round(ct, 2), round(total_reward, 4),
                             round(time.time() - t0, 2)])
        fcsv.flush()
        round_num += 1
        if round_num % args.save_every == 0 or round_num == 1:
            for trainer in trainers:
                trainer.save_model(out_dir)
        elapsed = time.time() - start
        done_eps = sum(t.episode_count for t in trainers)
        total_eps = args.episodes * len(trainers)
        print(f"轮次{round_num:4d} | 进度 {done_eps}/{total_eps} "
              f"({100*done_eps/total_eps:.1f}%) | 已用时 {elapsed/60:.1f}min | "
              f"本轮均值 {sum(t.completion_times[-1] for t in trainers)/len(trainers):.1f}",
              flush=True)

    for trainer in trainers:
        path = trainer.save_model(out_dir)
        print(f"{trainer.name}: 模型已保存 {path}, 全程最优 {trainer.best_completion_time:.1f}")
    fcsv.close()
    print(f"训练结束, 总用时 {(time.time()-start)/60:.1f}min, 曲线见 {curves_csv}")


if __name__ == '__main__':
    main()
