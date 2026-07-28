# -*- coding: utf-8 -*-
"""
D1-D12顺序轮询训练主程序。

训练方式: 单一共享的策略/价值网络, 全局共训练1200个episode(可配置),
每个episode按D1→D2→...→D12→D1→...的顺序切换到下一个算例
(即每个算例各训练 total_episodes/12 = 100 个episode)。

训练曲线逐episode追加写入 results/train_curves.csv;
共享模型定期保存到 results/ppo_ood_shared.pth, 支持 --resume 断点续训;
创建 ood_experiments/STOP.txt 可随时安全终止。

用法:
  python ood_experiments/run_training.py --total-episodes 1200
  python ood_experiments/run_training.py --resume          # 从上次断点继续
最终结果由 collect_results.py 基于每个算例最后阶段的收敛窗口统计。
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.env import FAJSP_Environment  # noqa: E402
from ood_experiments.configs import OOD_CONFIGS, ORDER  # noqa: E402
from ood_experiments.ppo_headless import build_shared_agent, run_episode  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, 'ood_experiments', 'results')
CURVES_CSV = os.path.join(RESULT_DIR, 'train_curves.csv')
CHECKPOINT = os.path.join(RESULT_DIR, 'ppo_ood_shared.pth')
STOP_FILE = os.path.join(REPO_ROOT, 'ood_experiments', 'STOP.txt')
CSV_FIELDS = ['global_episode', 'instance', 'episode', 'makespan', 'reward', 'seconds']


def load_resume_state():
    """从曲线CSV恢复全局episode计数与各算例episode计数"""
    start_episode = 0
    inst_counts = defaultdict(int)
    if os.path.exists(CURVES_CSV):
        with open(CURVES_CSV) as f:
            for row in csv.DictReader(f):
                start_episode = max(start_episode, int(row['global_episode']) + 1)
                inst_counts[row['instance']] += 1
    return start_episode, inst_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-episodes', type=int, default=1200,
                        help='全局训练episode总数(顺序轮询分摊到各算例)')
    parser.add_argument('--instances', type=str, default=','.join(ORDER),
                        help='参与轮询的算例, 逗号分隔, 默认D1-D12全部')
    parser.add_argument('--save-every', type=int, default=12,
                        help='每多少个episode保存一次共享模型')
    parser.add_argument('--resume', action='store_true',
                        help='从results/下的曲线与模型断点续训')
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    os.makedirs(RESULT_DIR, exist_ok=True)

    names = [n.strip() for n in args.instances.split(',') if n.strip()]
    print(f"顺序轮询算例: {names}")
    envs = {}
    for name in names:
        t0 = time.time()
        env = FAJSP_Environment(**OOD_CONFIGS[name]['env_kwargs'])
        env.reset()
        envs[name] = env
        print(f"  {name} 就绪 ({time.time()-t0:.1f}s), "
              f"工序类型{len(env.kind_task_tuple)}, 流体下界{env.fluid_completed_time:.1f}")

    # 共享agent(维度对所有算例一致)
    agent = build_shared_agent(envs[names[0]])

    start_episode, inst_counts = 0, defaultdict(int)
    best_ct = defaultdict(lambda: float('inf'))
    if args.resume:
        start_episode, inst_counts = load_resume_state()
        if os.path.exists(CHECKPOINT):
            ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
            agent.policy_net.load_state_dict(ckpt['policy'])
            agent.value_net.load_state_dict(ckpt['value'])
            agent.old_policy_net.load_state_dict(ckpt['policy'])
            best_ct.update(ckpt.get('best_completion_times', {}))
            print(f"断点续训: 从全局episode {start_episode} 继续")
        else:
            print("未找到模型断点, 从头开始(曲线计数沿用已有CSV)")

    new_file = not os.path.exists(CURVES_CSV)
    fcsv = open(CURVES_CSV, 'a', newline='')
    writer = csv.writer(fcsv)
    if new_file:
        writer.writerow(CSV_FIELDS)

    def save_checkpoint():
        torch.save({
            'policy': agent.policy_net.state_dict(),
            'value': agent.value_net.state_dict(),
            'best_policy': agent.best_policy_state,
            'best_value': agent.best_value_state,
            'best_completion_times': dict(best_ct),
            'instances': names,
            'env_kwargs': {n: OOD_CONFIGS[n]['env_kwargs'] for n in names},
        }, CHECKPOINT)

    start_time = time.time()
    for e in range(start_episode, args.total_episodes):
        if os.path.exists(STOP_FILE):
            print("\n检测到STOP.txt, 安全终止")
            break
        name = names[e % len(names)]
        env = envs[name]

        t0 = time.time()
        total_reward, ct, _, _, _ = run_episode(agent, env)
        seconds = time.time() - t0

        inst_ep = inst_counts[name]
        inst_counts[name] += 1
        writer.writerow([e, name, inst_ep, round(ct, 2),
                         round(total_reward, 4), round(seconds, 2)])
        fcsv.flush()

        if ct < best_ct[name]:
            best_ct[name] = ct
            agent.update_best_params(ct)

        if (e + 1) % args.save_every == 0:
            save_checkpoint()
        if (e + 1) % len(names) == 0:  # 每完成一轮打印进度
            elapsed = time.time() - start_time
            done = e + 1 - start_episode
            remain = (args.total_episodes - e - 1) * elapsed / max(done, 1)
            print(f"episode {e+1:5d}/{args.total_episodes} | {name} makespan={ct:.0f} | "
                  f"已用时{elapsed/60:.1f}min 预计剩余{remain/60:.0f}min", flush=True)

    save_checkpoint()
    fcsv.close()
    print(f"\n训练结束, 总用时 {(time.time()-start_time)/60:.1f}min")
    print(f"各算例最优完工时间: {dict(best_ct)}")
    print(f"曲线: {CURVES_CSV}\n模型: {CHECKPOINT}")
    print("运行 python ood_experiments/collect_results.py 生成最终汇总表")


if __name__ == '__main__':
    main()
