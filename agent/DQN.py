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
from collections import deque

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# python -m visdom.server
# 初始化Visdom
viz = visdom.Visdom(env='FAJSP_DQN_Training')
print("Visdom可视化已启动，访问: http://localhost:8097")


#########################################################
# DQN 网络：CNN 状态特征 + ATN 动作特征 + MLP Q值
#########################################################
class DQNetwork(nn.Module):
    def __init__(self, action_feature_dim, combined_dim):
        super(DQNetwork, self).__init__()
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

        # Compute Q value for each action
        q_values = []
        for i in range(len(candidate_actions)):
            af = action_trans[:, i, :]
            combined = torch.cat([state_feature, af], dim=1)
            q_value = self.mlp(combined)
            q_values.append(q_value)

        q_values_tensor = torch.cat(q_values, dim=1).squeeze(0)
        return q_values_tensor


#########################################################
# DQN Agent
#########################################################
class DQNAgent:
    def __init__(self, action_feature_dim, combined_dim,
                 lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.97,
                 memory_size=5000, batch_size=32, target_update_freq=50):

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # 衰减率
        self.batch_size = batch_size  # 批量大小
        self.target_update_freq = target_update_freq  # 目标网络更新频率
        self.device = device

        # 主网络
        self.q_net = DQNetwork(action_feature_dim, combined_dim).to(self.device)
        # 目标网络
        self.target_q_net = DQNetwork(action_feature_dim, combined_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)

        self.train_step_counter = 0

    #########################################################
    # 选择动作（epsilon-greedy策略）
    #########################################################
    def select_action(self, state, candidate_actions, candidate_actions_features):
        if len(candidate_actions) == 0:
            return None, None, None

        # epsilon-greedy策略
        if random.random() < self.epsilon:
            # 随机选择动作
            idx = random.randint(0, len(candidate_actions) - 1)
            action = candidate_actions[idx]
            q_value = None
        else:
            # 贪婪选择动作
            with torch.no_grad():
                q_values = self.q_net(state, candidate_actions, candidate_actions_features)
                idx = torch.argmax(q_values).item()
                action = candidate_actions[idx]
                q_value = q_values[idx].item()

        return action, q_value, candidate_actions_features[action]

    #########################################################
    # 存储经验
    #########################################################
    def store_transition(self, state, action, reward, next_state, done,
                         action_features, candidate_actions, candidate_actions_features,
                         next_candidate_actions=None, next_candidate_actions_features=None):
        """存储经验，包括下一状态的候选动作"""

        self.memory.append({
            'state': state,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'done': float(done),
            'action_features': action_features,
            'candidate_actions': list(candidate_actions),
            'candidate_actions_features': dict(candidate_actions_features),
            # 添加下一状态信息
            'next_candidate_actions': list(next_candidate_actions) if next_candidate_actions else [],
            'next_candidate_actions_features': dict(
                next_candidate_actions_features) if next_candidate_actions_features else {}
        })

    #########################################################
    # 训练步骤
    #########################################################
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0

        # 从经验回放中采样
        batch = random.sample(self.memory, self.batch_size)

        # 计算当前Q值
        current_q_list = []
        target_q_list = []

        for i in range(self.batch_size):
            sample = batch[i]

            # 计算当前Q值
            q_values = self.q_net(sample['state'],
                                  sample['candidate_actions'],
                                  sample['candidate_actions_features'])
            action_idx = sample['candidate_actions'].index(sample['action'])
            current_q = q_values[action_idx].unsqueeze(0)
            current_q_list.append(current_q)

            # 计算目标Q值（使用目标网络）
            with torch.no_grad():
                if sample['done'] or not sample['next_candidate_actions']:
                    # 终止状态或没有候选动作
                    target_q = torch.tensor(sample['reward']).float().to(self.device)
                else:
                    # 非终止状态，计算max_a' Q_target(s', a')
                    next_q_values = self.target_q_net(
                        sample['next_state'],
                        sample['next_candidate_actions'],
                        sample['next_candidate_actions_features']
                    )
                    max_next_q = torch.max(next_q_values)
                    target_q = sample['reward'] + self.gamma * max_next_q

                target_q_list.append(target_q.unsqueeze(0))

        current_q_values = torch.cat(current_q_list)
        target_q_values = torch.cat(target_q_list)

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()


        # 更新目标网络
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

        # 一个周期进行一次epsilon衰减
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


#########################################################
# DQN Trainer
#########################################################
class DQNTrainer:
    def __init__(self, data_config, viz_windows):
        self.data_config = data_config
        self.config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"
        self.viz_windows = viz_windows

        # 创建环境和DQN Agent
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

        # 创建DQN Agent
        self.agent = DQNAgent(
            action_feature_dim=feature_dim,
            combined_dim=state_dim + feature_dim
        )

        # 记录数据
        self.episode_rewards = []
        self.completion_times = []
        self.losses = []
        self.epsilons = []

        # 新增：记录每个决策步的时间
        self.decision_times = []

        self.episode_count = 0
        self.best_completion_time = float('inf')
        self.best_schedule_log = None

    def train_one_episode(self):
        """训练一个周期（修复版本）"""
        self.env.reset()
        state = self.env.current_state
        total_reward = 0
        loss = 0


        while True:
            # 当前状态的候选动作

            candidate_actions = self.env.get_candidate_actions()

            if not candidate_actions:
                break

            candidate_actions_features = {
                a: self.env.action_features(a)[a]
                for a in candidate_actions
            }

            # 选择动作（计时开始）
            decision_start = time.time()
            action, q_value, act_feat = self.agent.select_action(
                state, candidate_actions, candidate_actions_features
            )


            # 执行动作，获取下一状态
            next_state, reward, done = self.env.step(action)

            # 获取下一状态的候选动作（如果是终止状态则为空）
            if done:
                next_candidate_actions = []
                next_candidate_actions_features = {}
            else:
                next_candidate_actions = self.env.get_candidate_actions()
                next_candidate_actions_features = {
                    a: self.env.action_features(a)[a]
                    for a in next_candidate_actions
                }

            # 存储经验（包括下一状态信息）
            self.agent.store_transition(
                state, action, reward, next_state, done,
                act_feat, candidate_actions, candidate_actions_features,
                next_candidate_actions, next_candidate_actions_features  # 新增参数
            )

            # 训练
            loss = self.agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break



        self.agent.decay_epsilon()
        # 记录数据
        self.episode_rewards.append(total_reward)
        self.completion_times.append(self.env.completion_time)
        self.losses.append(loss if loss > 0 else 0)
        self.epsilons.append(self.agent.epsilon)

        current_ct = float(self.env.completion_time)

        # 更新最优记录
        if current_ct < self.best_completion_time:
            self.best_completion_time = current_ct
            self.best_schedule_log = copy.deepcopy(getattr(self.env, "schedule_log", []))

        # 更新Visdom - 类似PPO的方式
        if self.episode_count == 0:
            # 第一个周期：替换掉初始的0值

            viz.line(
                X=np.array([self.episode_count]),
                Y=np.array([current_ct]),
                win=self.viz_windows['ct_win'],
                update='replace',
                opts=dict(
                    title=f'Completion Time - {self.config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time',
                    ytickmin=current_ct * 0.9,
                    ytickmax=current_ct * 1.1
                )
            )
        else:

            viz.line(
                X=np.array([self.episode_count]),
                Y=np.array([current_ct]),
                win=self.viz_windows['ct_win'],
                update='append',
                opts=dict(title=f'Completion Time - {self.config_str}', xlabel='Episode', ylabel='Completion Time')
            )

        self.episode_count += 1
        return total_reward, self.env.completion_time, loss, self.agent.epsilon

    def save_results(self):
        """保存训练结果"""
        config_str = self.config_str
        save_path = f"dqn_fajsp_model_{config_str}.pth"

        # 获取环境中的任务信息用于甘特图
        kind_task_tuple = getattr(self.env, 'kind_task_tuple', [])
        kind_task_tuple_r = getattr(self.env, 'kind_task_tuple_r', [])
        kind_task_tuple_a = getattr(self.env, 'kind_task_tuple_a', [])
        machine_tuple = getattr(self.env, 'machine_tuple', [])

        torch.save({
            'config': self.data_config,
            'q_net': self.agent.q_net.state_dict(),
            'target_q_net': self.agent.target_q_net.state_dict(),
            'episode_rewards': self.episode_rewards,
            'completion_times': self.completion_times,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'best_completion_time': self.best_completion_time,
            # 添加甘特图所需数据
            'best_schedule_log': self.best_schedule_log,
            'gantt_info': {
                'kind_task_tuple': kind_task_tuple,
                'kind_task_tuple_r': kind_task_tuple_r,
                'kind_task_tuple_a': kind_task_tuple_a,
                'machine_tuple': machine_tuple
            },
        }, save_path)

        print(f"✅ {config_str}: DQN模型已保存, 最优完工时间: {self.best_completion_time:.2f}")
        return save_path



#########################################################
# 轮训训练主程序（类似PPO）
#########################################################
def train_round_robin_dqn(data_configs, total_episodes_per_instance=500):
    """轮训训练：每个算例训练1个周期，然后切换到下一个"""

    # 计算总训练周期数
    total_episodes = len(data_configs) * total_episodes_per_instance

    # 为每个算例创建visdom窗口
    trainers = []
    for data_config in data_configs:
        config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"

        # 创建窗口时使用一个隐藏的初始点
        viz_windows = {
            'ct_win': viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(
                    title=f'Completion Time - {config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time',
                    env='FAJSP_DQN_Training',
                    showlegend=False
                )
            )
        }

        trainer = DQNTrainer(data_config, viz_windows)
        trainers.append(trainer)

    # 创建平均完工时间的Visdom窗口
    avg_ct_win = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Round Average Completion Time (DQN)',
            xlabel='Round',
            ylabel='Average CT',
            env='FAJSP_DQN_Training',
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

    try:
        while True:
            # 检查是否所有算例都训练完成
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
                total_reward, current_ct, loss, epsilon = trainer.train_one_episode()

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
                    opts=dict(title='Round Average Completion Time (DQN)', xlabel='Round', ylabel='Average CT')
                )

                # 更新进度条（在轮次结束后）
                progress = min(100, global_episode / total_episodes * 100)
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r全局进度: {bar} {progress:.1f}% ({global_episode}/{total_episodes})", end='', flush=True)

                round_num += 1

    except KeyboardInterrupt:
        print("\n\n❌ 训练被用户中断 (Ctrl+F2)")

    # 保存所有结果
    for trainer in trainers:
        trainer.save_results()

    total_time = time.time() - start_time



    print("\n" + "=" * 60)
    print(f"DQN轮训训练完成！总用时: {total_time:.1f}秒")
    print("=" * 60)


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
    # python -m visdom.server
    # 测试用的小列表
    test_list = [[8, 4, 2, 4, 2, 2]]

    print("=" * 60)
    print("DQN 轮训训练开始")
    print("按 Ctrl+F2 可中断训练")
    print("=" * 60)

    # 开始轮训训练
    trainers = train_round_robin_dqn(test_list, total_episodes_per_instance=1000)