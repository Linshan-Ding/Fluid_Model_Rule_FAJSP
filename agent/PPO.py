import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .CNN import CNNFeatureExtractor
from .ATN import SelfAttention
from .MLP import MLP
from env.env import FAJSP_Environment
import numpy as np
import random
import time
import copy
import visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# python -m visdom.server
# 初始化Visdom
viz = visdom.Visdom(env='FAJSP_PPO_Training')
print("Visdom可视化已启动，访问: http://localhost:8097")


#########################################################
# 策略网络：CNN 状态特征 + ATN 动作特征 + MLP 得分
#########################################################
class PolicyNetwork(nn.Module):
    def __init__(self, action_feature_dim, combined_dim):
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.cnn = CNNFeatureExtractor(input_channels=1).to(self.device)
        self.atn = SelfAttention(feature_dim=action_feature_dim, num_heads=1).to(self.device)
        self.mlp = MLP(input_dim=combined_dim).to(self.device)

    def forward(self, matrix_states, candidate_actions, candidate_actions_features):
        # CNN state features
        state_features = []
        for matrix in matrix_states:
            m = torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
            feat = self.cnn(m)
            state_features.append(feat)
        state_feature = torch.cat(state_features, dim=1)

        # ATN action features
        action_feature_list = []
        for a in candidate_actions:
            action_feature_list.append(candidate_actions_features[a])
        action_tensor = torch.tensor(action_feature_list).unsqueeze(0).float().to(self.device)
        action_trans = self.atn(action_tensor)

        # Score each action
        scores = []
        for i in range(len(candidate_actions)):
            af = action_trans[:, i, :]
            combined = torch.cat([state_feature, af], dim=1)
            score = self.mlp(combined)
            scores.append(score)

        scores_tensor = torch.cat(scores, dim=1).squeeze(0)
        action_probs = F.softmax(scores_tensor, dim=0)
        return action_probs


#########################################################
# Value 网络
#########################################################
class ValueNetwork(nn.Module):
    def __init__(self, mlp_value_dim):
        super(ValueNetwork, self).__init__()
        self.device = device

        self.cnn = CNNFeatureExtractor(input_channels=1).to(self.device)
        self.mlp = MLP(input_dim=mlp_value_dim).to(self.device)

    def forward(self, matrix_states):
        feats = []
        for matrix in matrix_states:
            m = torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
            feat = self.cnn(m)
            feats.append(feat)
        state_feature = torch.cat(feats, dim=1)
        return self.mlp(state_feature)


#########################################################
# PPO Agent
#########################################################
class PPO:
    def __init__(self, action_feature_dim, combined_dim, mlp_value_dim,
                 lr_policy=3e-4, lr_value=3e-4, gamma=1,
                 eps_clip=0.2, K_epochs=4, batch_size=32):

        # 共享模型输入维度，用于多算例一致性检查。
        self.action_feature_dim = action_feature_dim
        self.state_feature_dim = mlp_value_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.device = device

        self.policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.value_net = ValueNetwork(mlp_value_dim).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': lr_policy},
            {'params': self.value_net.parameters(), 'lr': lr_value}
        ])

        self.old_policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        self.memory = []

        # 添加最优参数记录
        self.best_policy_state = None
        self.best_value_state = None
        self.best_completion_time = float('inf')

    #########################################################
    # 更新最优参数
    #########################################################
    def update_best_params(self, completion_time):
        if completion_time < self.best_completion_time:
            self.best_completion_time = completion_time
            self.best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.best_value_state = copy.deepcopy(self.value_net.state_dict())
            return True
        return False

    #########################################################
    # 选择动作
    #########################################################
    def select_action(self, state, candidate_actions, candidate_actions_features):
        with torch.no_grad():
            probs = self.old_policy_net(state, candidate_actions, candidate_actions_features)

            if len(candidate_actions) == 0:
                return None, None, None, None

            # 依概率采样动作
            idx = torch.multinomial(probs, 1).item()
            action = candidate_actions[idx]
            log_p = torch.log(probs[idx] + 1e-10)

            return action, log_p, probs[idx], candidate_actions_features[action]

    #########################################################
    # 存储经验
    #########################################################
    def store_transition(self, state, action, log_prob,
                         reward, next_state, done,
                         action_features, candidate_actions, candidate_actions_features):

        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob.clone(),
            'reward': float(reward),
            'next_state': next_state,
            'done': float(done),
            'action_features': action_features,
            'candidate_actions': list(candidate_actions),
            'candidate_actions_features': dict(candidate_actions_features)
        })

    #########################################################
    # 计算 Advantage
    #########################################################
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        gamma, lam = self.gamma, 0.95

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * gae * (1 - dones[t])
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    #########################################################
    # PPO 更新
    #########################################################
    def update(self):
        if len(self.memory) < self.batch_size:
            return 0, 0

        states = [m['state'] for m in self.memory]
        actions = [m['action'] for m in self.memory]
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory]).to(self.device)
        rewards = torch.tensor([m['reward'] for m in self.memory]).float().to(self.device)
        dones = torch.tensor([m['done'] for m in self.memory]).float().to(self.device)
        next_states = [m['next_state'] for m in self.memory]

        # 计算 Value
        with torch.no_grad():
            values = torch.tensor([self.value_net(s).item() for s in states]).to(self.device)
            next_values = torch.tensor([
                self.value_net(ns).item() if ns is not None else 0.0 for ns in next_states
            ]).to(self.device)

        advantages = self.compute_advantages(rewards, values, next_values, dones)
        returns = advantages + values

        # 归一化
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses = [], []

        for _ in range(self.K_epochs):
            batch_indices = list(range(len(self.memory)))
            random.shuffle(batch_indices)

            for start in range(0, len(batch_indices), self.batch_size):
                end = start + self.batch_size
                idxs = batch_indices[start:end]
                if len(idxs) == 0:
                    continue

                batch_old = old_log_probs[idxs]
                batch_adv = advantages[idxs]
                batch_ret = returns[idxs]

                new_log_probs = []
                value_preds = []

                for bi, mem_idx in enumerate(idxs):
                    m = self.memory[mem_idx]

                    state = m['state']
                    ca = m['candidate_actions']
                    caf = m['candidate_actions_features']
                    act = m['action']

                    probs = self.policy_net(state, ca, caf)
                    act_index = ca.index(act)
                    new_log = torch.log(probs[act_index] + 1e-10)

                    new_log_probs.append(new_log)
                    value_preds.append(self.value_net(state))

                new_log_probs = torch.stack(new_log_probs).to(self.device)
                value_preds = torch.stack(value_preds).squeeze().to(self.device)

                ratios = torch.exp(new_log_probs - batch_old)

                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_preds, batch_ret)

                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.memory = []
        return np.mean(policy_losses), np.mean(value_losses)


#########################################################
# 单个算例的训练器
#########################################################
class InstanceTrainer:
    def __init__(self, data_config, viz_windows, shared_agent=None):
        self.data_config = data_config
        self.config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"
        self.viz_windows = viz_windows

        # 创建环境和PPO
        M_p, M_a, product_count, kind_count, J_r, N_p = data_config

        self.env = FAJSP_Environment(
            M_p=M_p, M_a=M_a,
            kind_count=kind_count,
            product_count=product_count,
            J_r=J_r, N_p=N_p
        )
        self.env.reset()

        # 获取动作特征维度
        ca = self.env.get_candidate_actions()
        if len(ca) == 0:
            raise ValueError("错误：没有候选动作")

        caf = self.env.action_features(ca[0])
        feature_dim = len(caf[ca[0]])

        # 获取 CNN 状态维度
        cnn = CNNFeatureExtractor(input_channels=1).to(device)
        s = self.env.current_state
        feats = []
        for m in s:
            t = torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)
            feats.append(cnn(t))
        sf = torch.cat(feats, dim=1)
        state_dim = sf.shape[1]

        # 第一个 Trainer 创建 PPO；之后所有 Trainer 共用同一 PPO 对象。
        if shared_agent is None:
            self.agent = PPO(
                action_feature_dim=feature_dim,
                combined_dim=state_dim + feature_dim,
                mlp_value_dim=state_dim
            )
        else:
            if (shared_agent.action_feature_dim != feature_dim or
                    shared_agent.state_feature_dim != state_dim):
                raise ValueError(
                    f"{self.config_str} 的特征维度不匹配："
                    f"action={feature_dim}, state={state_dim}; "
                    f"shared action={shared_agent.action_feature_dim}, "
                    f"shared state={shared_agent.state_feature_dim}"
                )
            self.agent = shared_agent

        # 记录数据
        self.episode_rewards = []
        self.completion_times = []
        self.policy_losses = []
        self.value_losses = []

        self.episode_count = 0

    def train_one_episode(self):
        """训练一个周期：每个 Episode 都按当前参数重新生成实例。"""
        # 不传 rng_seed；随机种子是否固定由 Instance_generate.py 决定。
        # 只 reset() 会重复使用旧算例，因此这里必须重新创建环境。
        M_p, M_a, product_count, kind_count, J_r, N_p = self.data_config
        self.env = FAJSP_Environment(
            M_p=M_p, M_a=M_a,
            kind_count=kind_count, product_count=product_count,
            J_r=J_r, N_p=N_p
        )
        self.env.reset()
        state = self.env.current_state
        total_reward = 0
        while 1:
            candidate_actions = self.env.get_candidate_actions()
            if not candidate_actions:
                break

            candidate_actions_features = {
                a: self.env.action_features(a)[a]
                for a in candidate_actions
            }

            action, logp, _, act_feat = \
                self.agent.select_action(state, candidate_actions, candidate_actions_features)

            next_state, reward, done = self.env.step(action)

            self.agent.store_transition(
                state, action, logp, reward, next_state, done,
                act_feat, candidate_actions, candidate_actions_features
            )

            state = next_state
            total_reward += reward
            if done:
                break

        # PPO更新
        pol_loss, val_loss = self.agent.update()

        # 记录数据
        self.episode_rewards.append(total_reward)
        self.completion_times.append(self.env.completion_time)
        self.policy_losses.append(pol_loss)
        self.value_losses.append(val_loss)

        current_ct = float(self.env.completion_time)

        # 更新Visdom - 关键修改
        if self.episode_count == 0:
            # 第一个周期：替换掉初始的0值
            # viz.line(
            #     X=np.array([self.episode_count]),
            #     Y=np.array([total_reward]),
            #     win=self.viz_windows['reward_win'],
            #     update='replace',  # 关键：替换而不是追加
            #     opts=dict(
            #         title=f'Reward - {self.config_str}',
            #         xlabel='Episode',
            #         ylabel='Reward',
            #         ytickmin=total_reward * 0.9,  # Y轴从第一个值的90%开始
            #         ytickmax=total_reward * 1.1  # 到第一个值的110%
            #     )
            # )

            viz.line(
                X=np.array([self.episode_count]),
                Y=np.array([current_ct]),
                win=self.viz_windows['ct_win'],
                update='replace',  # 关键：替换而不是追加
                opts=dict(
                    title=f'Completion Time - {self.config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time',
                    ytickmin=current_ct * 0.9,  # Y轴从第一个值的90%开始
                    ytickmax=current_ct * 1.1  # 到第一个值的110%
                )
            )
        else:
            # 后续周期正常追加
            # viz.line(
            #     X=np.array([self.episode_count]),
            #     Y=np.array([total_reward]),
            #     win=self.viz_windows['reward_win'],
            #     update='append',
            #     opts=dict(title=f'Reward - {self.config_str}', xlabel='Episode', ylabel='Reward')
            # )

            viz.line(
                X=np.array([self.episode_count]),
                Y=np.array([current_ct]),
                win=self.viz_windows['ct_win'],
                update='append',
                opts=dict(title=f'Completion Time - {self.config_str}', xlabel='Episode', ylabel='Completion Time')
            )

        # 损失函数图表（第一个周期也替换）
        # if self.episode_count == 0:
        #     viz.line(
        #         X=np.array([self.episode_count]),
        #         Y=np.array([pol_loss]),
        #         win=self.viz_windows['policy_loss_win'],
        #         update='replace',
        #         opts=dict(title=f'Policy Loss - {self.config_str}', xlabel='Episode', ylabel='Loss')
        #     )
        #
        #     viz.line(
        #         X=np.array([self.episode_count]),
        #         Y=np.array([val_loss]),
        #         win=self.viz_windows['value_loss_win'],
        #         update='replace',
        #         opts=dict(title=f'Value Loss - {self.config_str}', xlabel='Episode', ylabel='Loss')
        #     )
        # else:
        #     viz.line(
        #         X=np.array([self.episode_count]),
        #         Y=np.array([pol_loss]),
        #         win=self.viz_windows['policy_loss_win'],
        #         update='append',
        #         opts=dict(title=f'Policy Loss - {self.config_str}', xlabel='Episode', ylabel='Loss')
        #     )
        #
        #     viz.line(
        #         X=np.array([self.episode_count]),
        #         Y=np.array([val_loss]),
        #         win=self.viz_windows['value_loss_win'],
        #         update='append',
        #         opts=dict(title=f'Value Loss - {self.config_str}', xlabel='Episode', ylabel='Loss')
        #     )

        self.episode_count += 1

        return total_reward, current_ct, pol_loss, val_loss



#########################################################
# 随机抽样训练主程序
#########################################################
def train_random_sampling(data_configs, total_episodes=1000,
                          output_dir="shared_training_results"):
    """每个 Episode 从配置表有放回随机抽取一组，并重新生成该组的算例。"""
    if not data_configs:
        raise ValueError("data_configs 不能为空")
    if total_episodes < 1:
        raise ValueError("total_episodes 必须是正整数")
    os.makedirs(output_dir, exist_ok=True)

    # 每种参数有自己的训练统计和 Visdom 曲线；所有配置共享同一个 PPO。
    trainers = []
    shared_agent = None
    for data_config in data_configs:
        config_str = "_".join(map(str, data_config))
        viz_windows = {
            'ct_win': viz.line(
                X=np.array([0]), Y=np.array([0]),
                opts=dict(
                    title=f'Completion Time - {config_str}',
                    xlabel='Episode', ylabel='Completion Time',
                    env='FAJSP_PPO_Training', showlegend=False
                )
            )
        }
        trainer = InstanceTrainer(data_config, viz_windows,
                                  shared_agent=shared_agent)
        if shared_agent is None:
            shared_agent = trainer.agent
        trainers.append(trainer)

    start_time = time.time()
    sampled_episodes = []
    bar_length = 40
    print(f"从 {len(data_configs)} 组配置中有放回随机抽样，计划训练 {total_episodes} 个 Episode")

    # 无轮次、无顺序遍历：每个 Episode 独立调用一次 random.choice。
    for episode in range(1, total_episodes + 1):
        if os.path.exists('STOP.txt'):
            print("\n检测到 STOP.txt，提前结束训练")
            break
        trainer = random.choice(trainers)
        # 内部重新创建 FAJSP_Environment，而不是只调用 reset()。
        _, current_ct, _, _ = trainer.train_one_episode()
        sampled_episodes.append((episode, trainer.config_str, float(current_ct)))

        progress = 100 * episode / total_episodes
        filled = int(bar_length * episode / total_episodes)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r全局进度: {bar} {progress:.1f}% ({episode}/{total_episodes})",
              end='', flush=True)
    print()

    # 不生成每组配置对应的模型文件；只保存抽样记录和最终共享模型。
    import csv
    with open(os.path.join(output_dir, 'ppo_episode_sampling.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'config', 'makespan'])
        writer.writerows(sampled_episodes)

    # 不跨规模比较 Makespan；仅保存最终共享权重。
    # 保存实际完成训练后的同一套 Actor、Critic 和优化器。
    torch.save({
        'policy': shared_agent.policy_net.state_dict(),
        'value': shared_agent.value_net.state_dict(),
        'optimizer': shared_agent.optimizer.state_dict(),
        'training_mode': 'random',
        'configs': data_configs,
        'episodes_trained': len(sampled_episodes),
    }, os.path.join(output_dir, 'ppo_fajsp_final.pth'))

    print(f"随机抽样训练结束，完成 {len(sampled_episodes)} 个 Episode，"
          f"总用时 {time.time() - start_time:.1f} 秒")
    return trainers


if __name__ == "__main__":
    # 所有算例配置
    data_list = [
        [4, 2, 2, 4, 2, 2], [4, 2, 2, 4, 2, 4], [4, 2, 2, 4, 2, 6],
        [4, 2, 4, 8, 4, 2], [4, 2, 4, 8, 4, 4], [4, 2, 4, 8, 4, 6],
        [4, 2, 6, 12, 6, 2], [4, 2, 6, 12, 6, 4], [4, 2, 6, 12, 6, 6],
        [8, 4, 2, 4, 2, 2], [8, 4, 2, 4, 2, 4], [8, 4, 2, 4, 2, 6],
        [8, 4, 4, 8, 4, 2], [8, 4, 4, 8, 4, 4], [8, 4, 4, 8, 4, 6],
        [8, 4, 6, 12, 6, 2], [8, 4, 6, 12, 6, 4], [8, 4, 6, 12, 6, 6],
        [12, 6, 2, 4, 2, 2], [12, 6, 2, 4, 2, 4], [12, 6, 2, 4, 2, 6],
        [12, 6, 4, 8, 4, 2], [12, 6, 4, 8, 4, 4], [12, 6, 4, 8, 4, 6],
        [12, 6, 6, 12, 6, 2], [12, 6, 6, 12, 6, 4], [12, 6, 6, 12, 6, 6]
    ]

    import argparse
    parser = argparse.ArgumentParser(description="PPO：27组参数随机抽样、每轮新建算例")
    parser.add_argument('--episodes', type=int, default=1000,
                        help='总 Episode 数（默认1000）')
    parser.add_argument('--output-dir', default='shared_training_results',
                        help='训练结果输出目录')
    args = parser.parse_args()
    train_random_sampling(data_list, total_episodes=args.episodes,
                          output_dir=args.output_dir)

    # 启动：python -m agent.PPO --episodes 1000
