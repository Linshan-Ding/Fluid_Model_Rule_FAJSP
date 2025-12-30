import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CNN import CNNFeatureExtractor
from ATN import SelfAttention
from MLP import MLP
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
                 lr_policy=3e-4, lr_value=1e-4, gamma=0.99,
                 eps_clip=0.2, K_epochs=10, batch_size=64):

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

        # 添加全局最优参数记录（用于轮训训练）
        self.global_best_policy_state = None
        self.global_best_value_state = None
        self.global_best_avg_ct = float('inf')
        self.global_best_round = -1

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
    # 更新全局最优参数（用于轮训训练）
    #########################################################
    def update_global_best_params(self, avg_completion_time, round_num):
        """更新全局最优参数（基于轮次平均完工时间）"""
        if avg_completion_time < self.global_best_avg_ct:
            self.global_best_avg_ct = avg_completion_time
            self.global_best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.global_best_value_state = copy.deepcopy(self.value_net.state_dict())
            self.global_best_round = round_num
            return True
        return False

    #########################################################
    # 选择动作（选择概率最高的动作）
    #########################################################
    def select_action(self, state, candidate_actions, candidate_actions_features):
        with torch.no_grad():
            probs = self.old_policy_net(state, candidate_actions, candidate_actions_features)

            if len(candidate_actions) == 0:
                return None, None, None, None

            # 选择概率最高的动作
            # idx = torch.argmax(probs).item()
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
    def __init__(self, data_config, viz_windows):
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

        # 创建PPO Agent
        self.agent = PPO(
            action_feature_dim=feature_dim,
            combined_dim=state_dim + feature_dim,
            mlp_value_dim=state_dim
        )

        # 记录数据
        self.episode_rewards = []
        self.completion_times = []
        self.policy_losses = []
        self.value_losses = []

        self.episode_count = 0
        self.best_completion_time = float('inf')
        self.best_schedule_log = None

    def train_one_episode(self):
        """训练一个周期"""
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

        # 更新最优记录
        if current_ct < self.best_completion_time:
            self.best_completion_time = current_ct
            self.best_schedule_log = copy.deepcopy(getattr(self.env, "schedule_log", []))
            self.agent.update_best_params(current_ct)

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

    def save_results(self):
        """保存训练结果"""
        config_str = self.config_str
        save_path = f"ppo_fajsp_model_{config_str}.pth"

        # 获取环境中的任务信息用于甘特图
        kind_task_tuple = getattr(self.env, 'kind_task_tuple', [])
        kind_task_tuple_r = getattr(self.env, 'kind_task_tuple_r', [])
        kind_task_tuple_a = getattr(self.env, 'kind_task_tuple_a', [])
        machine_tuple = getattr(self.env, 'machine_tuple', [])

        torch.save({
            'config': self.data_config,
            'policy': self.agent.best_policy_state,
            'value': self.agent.best_value_state,
            'reward': self.episode_rewards,
            'completion_times': self.completion_times,
            'best_completion_time': self.agent.best_completion_time,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            # 添加甘特图所需数据
            'best_schedule_log': self.best_schedule_log,
            'gantt_info': {
                'kind_task_tuple': kind_task_tuple,
                'kind_task_tuple_r': kind_task_tuple_r,
                'kind_task_tuple_a': kind_task_tuple_a,
                'machine_tuple': machine_tuple
            }
        }, save_path)

        print(f"✅ {config_str}: 模型已保存, 最优完工时间: {self.agent.best_completion_time:.2f}")
        return save_path


#########################################################
# 轮训训练主程序
#########################################################
def train_round_robin(data_configs, total_episodes_per_instance=10000):
    """轮训训练：每个算例训练1个周期，然后切换到下一个"""

    # 计算总训练周期数
    total_episodes = len(data_configs) * total_episodes_per_instance

    # 为每个算例创建visdom窗口 - 修改这里
    trainers = []
    for data_config in data_configs:
        config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"

        # 创建窗口时使用一个隐藏的初始点
        viz_windows = {
            # 'reward_win': viz.line(
            #     X=np.array([0]),  # Visdom要求至少有一个点
            #     Y=np.array([0]),  # 初始值设为0，但后面会覆盖
            #     opts=dict(
            #         title=f'Reward - {config_str}',
            #         xlabel='Episode',
            #         ylabel='Reward',
            #         env='FAJSP_PPO_Training',
            #         showlegend=False  # 不显示图例
            #     )
            # ),
            'ct_win': viz.line(
                X=np.array([0]),  # Visdom要求至少有一个点
                Y=np.array([0]),  # 初始值设为0，但后面会覆盖
                opts=dict(
                    title=f'Completion Time - {config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time',
                    env='FAJSP_PPO_Training',
                    showlegend=False  # 不显示图例
                )
            )
            # 'policy_loss_win': viz.line(
            #     X=np.array([0]),  # Visdom要求至少有一个点
            #     Y=np.array([0]),  # 初始值设为0，但后面会覆盖
            #     opts=dict(
            #         title=f'Policy Loss - {config_str}',
            #         xlabel='Episode',
            #         ylabel='Loss',
            #         env='FAJSP_PPO_Training',
            #         showlegend=False
            #     )
            # ),
            # 'value_loss_win': viz.line(
            #     X=np.array([0]),  # Visdom要求至少有一个点
            #     Y=np.array([0]),  # 初始值设为0，但后面会覆盖
            #     opts=dict(
            #         title=f'Value Loss - {config_str}',
            #         xlabel='Episode',
            #         ylabel='Loss',
            #         env='FAJSP_PPO_Training',
            #         showlegend=False
            #     )
            # )
        }

        trainer = InstanceTrainer(data_config, viz_windows)
        trainers.append(trainer)

    # 创建平均完工时间的Visdom窗口
    avg_ct_win = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Round Average Completion Time',
            xlabel='Round',
            ylabel='Average CT',
            env='FAJSP_PPO_Training',
            showlegend=False
        )
    )

    # 轮训训练
    start_time = time.time()
    global_episode = 0
    bar_length = 40

    # 添加轮次记录
    round_num = 0
    round_avg_ct_list = []

    # 显示初始进度条
    bar = '░' * bar_length
    print(f"全局进度: {bar} 0.0% (0/{total_episodes})", end='', flush=True)

    # 获取第一个trainer的agent引用（假设所有trainer共享同一个PPO结构）
    global_agent = trainers[0].agent if trainers else None

    while True:
        if os.path.exists("STOP.txt"):
            break

        all_done = all(t.episode_count >= total_episodes_per_instance for t in trainers)
        if all_done:
            print()
            break

        # 每个轮次开始前重置计数器
        round_total_ct = 0
        round_instance_count = 0

        # 轮流训练：每个算例训练1个周期
        for trainer in trainers:
            if trainer.episode_count >= total_episodes_per_instance:
                continue

            # 训练一个周期
            total_reward, current_ct, pol_loss, val_loss = trainer.train_one_episode()

            # 累计本轮次的总完工时间
            round_total_ct += current_ct
            round_instance_count += 1

            global_episode += 1

        # 计算本轮次的平均完工时间（在所有算例都训练完一个周期后）
        if round_instance_count > 0:
            round_avg_ct = round_total_ct / round_instance_count
            round_avg_ct_list.append(round_avg_ct)

            # 更新Visdom
            viz.line(
                X=np.array([round_num]),
                Y=np.array([round_avg_ct]),
                win=avg_ct_win,
                update='append',
                opts=dict(title='Round Average Completion Time', xlabel='Round', ylabel='Average CT')
            )

            # 更新全局最优参数
            if global_agent:
                updated = global_agent.update_global_best_params(round_avg_ct, round_num)

            # 更新进度条（在轮次结束后）
            progress = min(100, global_episode / total_episodes * 100)
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r全局进度: {bar} {progress:.1f}% ({global_episode}/{total_episodes})", end='', flush=True)

            round_num += 1

    # 保存所有结果
    for trainer in trainers:
        trainer.save_results()

    # 保存全局最优参数到单独的pth文件
    if global_agent and global_agent.global_best_policy_state is not None:
        global_best_path = "ppo_fajsp_global_best.pth"
        torch.save({
            'global_best_policy_state': global_agent.global_best_policy_state,
            'global_best_value_state': global_agent.global_best_value_state,
            'global_best_avg_ct': global_agent.global_best_avg_ct,
            'global_best_round': global_agent.global_best_round,
            'round_avg_ct_history': round_avg_ct_list
        }, global_best_path)

    total_time = time.time() - start_time
    print(f"轮训训练完成！总用时: {total_time:.1f}秒")

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

    # 测试用的小列表
    test_list = [[4, 2, 2, 4, 2, 2]]

    # 开始轮训训练
    trainers = train_round_robin(test_list, total_episodes_per_instance=100000)

    print("\n所有算例轮训训练完成！")