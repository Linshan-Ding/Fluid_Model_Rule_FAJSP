# 同步单线程A2C算法（优势演员-评论家算法）实现，包含GAE优势估计、n_step步更新，并支持轮训训练
import os
import sys
import signal
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

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化Visdom
viz = visdom.Visdom(env='FAJSP_A2C_Training')

# 全局中断标志
#检测到中断信号后需等待其完成一个轮次
interrupt_flag = False
def signal_handler(sig, frame):
    """处理中断信号（Ctrl+F2）"""
    global interrupt_flag
    interrupt_flag = True
    print("\n检测到中断信号 (Ctrl+F2)，正在保存当前状态...")

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

# 策略网络
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

# 价值网络
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

# ACAgent with GAE
class A2CAgent:
    def __init__(self, action_feature_dim, combined_dim, mlp_value_dim,
                 policy_lr=1e-3, value_lr=1e-3, gamma=0.99, gae_lambda=0.95, n_steps=5):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps # n步更新

        self.policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.value_net = ValueNetwork(mlp_value_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)

        self.transitions = []

        # 最优参数记录
        self.best_policy_state = None
        self.best_value_state = None
        self.best_completion_time = float('inf')

        # 全局最优参数记录（用于轮训训练）
        self.global_best_policy_state = None
        self.global_best_value_state = None
        self.global_best_avg_ct = float('inf')
        self.global_best_round = -1

    # 更新最优参数（基于单次完工时间）用于保存pth文件
    def update_best_params(self, completion_time):
        if completion_time < self.best_completion_time:
            self.best_completion_time = completion_time
            self.best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.best_value_state = copy.deepcopy(self.value_net.state_dict())
            return True
        return False

    # 更新全局最优参数（基于轮次平均完工时间）用于保存全局最优pth文件
    def update_global_best_params(self, avg_completion_time, round_num):
        """更新全局最优参数（基于轮次平均完工时间）"""
        if avg_completion_time < self.global_best_avg_ct:
            self.global_best_avg_ct = avg_completion_time
            self.global_best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.global_best_value_state = copy.deepcopy(self.value_net.state_dict())
            self.global_best_round = round_num
            return True
        return False

    def take_action(self, matrix_states, candidate_actions, candidate_actions_features):
        """选择动作 - 返回动作索引和状态价值"""
        with torch.no_grad():
            # 获取动作概率
            action_probs = self.policy_net(matrix_states, candidate_actions, candidate_actions_features)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_idx = action.item()

            # 获取状态价值
            state_value = self.value_net(matrix_states)
            if state_value.dim() > 0:
                state_value = state_value.squeeze()

        return action_idx, state_value.item()

    def store_transition(self, state, action_idx, state_value, reward, next_state, done,
                         candidate_actions, candidate_actions_features):
        """存储经验"""
        self.transitions.append({
            'states': state,
            'action_idx': action_idx,
            'state_value': state_value,
            'rewards': reward,
            'next_states': next_state,
            'dones': done,
            'candidate_actions': candidate_actions,
            'candidate_actions_features': candidate_actions_features
        })
    #与基础ACAgent不同，A2C使用GAE优势估计
    def compute_gae_advantages(self, rewards, values, next_values, dones):
        """计算GAE优势"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]

            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        return torch.stack(advantages)

    def update(self):
        """更新网络"""
        if len(self.transitions) == 0:
            return 0.0, 0.0

        # 1. 准备数据
        states = [t['states'] for t in self.transitions]
        action_idxs = [t['action_idx'] for t in self.transitions]
        rewards = torch.tensor([t['rewards'] for t in self.transitions],
                               dtype=torch.float32).to(self.device)
        next_states = [t['next_states'] for t in self.transitions]
        dones = torch.tensor([t['dones'] for t in self.transitions],
                             dtype=torch.float32).to(self.device)

        candidate_actions_list = [t['candidate_actions'] for t in self.transitions]
        candidate_actions_features_list = [t['candidate_actions_features'] for t in self.transitions]

        # 2. 计算当前状态价值
        state_values = []
        for s in states:
            value = self.value_net(s)
            if value.dim() > 0:
                value = value.squeeze()
            state_values.append(value)
        state_values = torch.stack(state_values)

        # 3. 计算下一个状态价值
        next_state_values = []
        for ns in next_states:
            with torch.no_grad():
                next_value = self.value_net(ns)
                if next_value.dim() > 0:
                    next_value = next_value.squeeze()
                next_state_values.append(next_value)
        next_state_values = torch.stack(next_state_values)

        # 4. 计算GAE优势
        advantages = self.compute_gae_advantages(
            rewards, state_values.detach(), next_state_values, dones
        )

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5. 计算回报（用于Critic训练）
        returns = advantages + state_values.detach()

        # 6. 重新计算log_probs以保持计算图
        log_probs = []
        for i in range(len(states)):
            action_probs = self.policy_net(
                states[i],
                candidate_actions_list[i],
                candidate_actions_features_list[i]
            )
            log_prob = torch.log(action_probs[action_idxs[i]] + 1e-10)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)

        # 7. 计算策略损失
        actor_loss = -torch.mean(log_probs * advantages)

        # 8. 计算价值损失
        critic_loss = F.mse_loss(state_values, returns.detach())

        # 9. 更新策略网络
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.policy_optimizer.step()

        # 10. 更新价值网络
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.value_optimizer.step()

        # 清空经验
        self.transitions.clear()

        return actor_loss.item(), critic_loss.item()

# 单个算例的训练器,用于轮训训练
class InstanceTrainer:
    def __init__(self, data_config, viz_windows, n_steps=5):
        self.data_config = data_config
        self.config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"
        self.viz_windows = viz_windows # Visdom窗口字典
        self.n_steps = n_steps

        # 创建环境和A2C Agent
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

        # 创建A2C Agent
        self.agent = A2CAgent(
            action_feature_dim=feature_dim,
            combined_dim=state_dim + feature_dim,
            mlp_value_dim=state_dim,
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=n_steps
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
        self.env.reset()
        state = self.env.current_state
        total_reward = 0

        while True:
            candidate_actions = self.env.get_candidate_actions()
            if not candidate_actions:
                break

            candidate_actions_features = {
                a: self.env.action_features(a)[a]
                for a in candidate_actions
            }

            # 收集n步经验,与基础ACAgent不同，AC一个周期进行一次进行更新，A2C则是每n步更新一次
            for step in range(self.n_steps):
                # 选择动作
                action_idx, state_value = self.agent.take_action(
                    state, candidate_actions, candidate_actions_features
                )
                action = candidate_actions[action_idx]

                # 执行动作
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # 存储经验
                self.agent.store_transition(
                    state=state,
                    action_idx=action_idx,
                    state_value=state_value,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    candidate_actions=candidate_actions,
                    candidate_actions_features=candidate_actions_features
                )

                state = next_state

                if done:
                    break

                # 更新候选动作
                candidate_actions = self.env.get_candidate_actions()
                if not candidate_actions:
                    break

                candidate_actions_features = {
                    a: self.env.action_features(a)[a]
                    for a in candidate_actions
                }

            # 更新网络
            actor_loss, critic_loss = self.agent.update()

            if done:
                break

        # 记录数据
        self.episode_rewards.append(total_reward)
        self.completion_times.append(self.env.completion_time)
        #可用于绘图
        self.policy_losses.append(actor_loss)
        self.value_losses.append(critic_loss)

        current_ct = float(self.env.completion_time)

        # 更新最优记录
        if current_ct < self.best_completion_time:
            self.best_completion_time = current_ct
            self.best_schedule_log = copy.deepcopy(getattr(self.env, "schedule_log", []))
            self.agent.update_best_params(current_ct)

        # 更新Visdom，不以0为基点绘图
        if self.episode_count == 0:
            viz.line(
                X=np.array([self.episode_count]),
                Y=np.array([current_ct]),
                win=self.viz_windows['ct_win'],
                update='replace',
                opts=dict(
                    title=f'Completion Time - {self.config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time'
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

        return total_reward, current_ct, actor_loss, critic_loss

    # 保存中断结果
    def save_interrupt_results(self, round_num=None):
        config_str = self.config_str
        interrupt_path = f"a2c_fajsp_model_{config_str}.pth"

        kind_task_tuple = getattr(self.env, 'kind_task_tuple', [])
        kind_task_tuple_r = getattr(self.env, 'kind_task_tuple_r', [])
        kind_task_tuple_a = getattr(self.env, 'kind_task_tuple_a', [])
        machine_tuple = getattr(self.env, 'machine_tuple', [])

        save_data = {
            'config': self.data_config,
            'episode_count': self.episode_count,
            'current_policy': self.agent.policy_net.state_dict(),
            'current_value': self.agent.value_net.state_dict(),
            'policy_optimizer': self.agent.policy_optimizer.state_dict(),
            'value_optimizer': self.agent.value_optimizer.state_dict(),
            'best_policy': self.agent.best_policy_state,
            'best_value': self.agent.best_value_state,
            'reward': self.episode_rewards,
            'completion_times': self.completion_times,
            'best_completion_time': self.agent.best_completion_time,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'best_schedule_log': self.best_schedule_log,
            'gantt_info': {
                'kind_task_tuple': kind_task_tuple,
                'kind_task_tuple_r': kind_task_tuple_r,
                'kind_task_tuple_a': kind_task_tuple_a,
                'machine_tuple': machine_tuple
            },
            'interrupt_round': round_num,
            'interrupt_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        torch.save(save_data, interrupt_path)
        return interrupt_path

    # 正常结束保存最终结果
    def save_results(self):
        config_str = self.config_str
        save_path = f"a2c_fajsp_model_{config_str}.pth"

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
            'best_schedule_log': self.best_schedule_log,
            'gantt_info': {
                'kind_task_tuple': kind_task_tuple,
                'kind_task_tuple_r': kind_task_tuple_r,
                'kind_task_tuple_a': kind_task_tuple_a,
                'machine_tuple': machine_tuple
            }
        }, save_path)

        return save_path

# 轮训训练主程序
def train_round_robin(data_configs, total_episodes_per_instance=10000, n_steps=5):
    global interrupt_flag

    # 计算总训练周期数
    total_episodes = len(data_configs) * total_episodes_per_instance

    # 为每个算例创建visdom窗口
    trainers = []
    for data_config in data_configs:
        config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"

        viz_windows = {
            'ct_win': viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(
                    title=f'Completion Time - {config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time',
                    env='FAJSP_A2C_Training',
                    showlegend=False
                )
            )
        }

        trainer = InstanceTrainer(data_config, viz_windows, n_steps=n_steps)
        trainers.append(trainer)

    # 创建平均完工时间的Visdom窗口，可有可不有
    avg_ct_win = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Round Average Completion Time (A2C)',
            xlabel='Round',
            ylabel='Average CT',
            env='FAJSP_A2C_Training',
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

    # 显示初始进度条，多轮次1更新
    bar = '░' * bar_length
    print(f"全局进度: {bar} 0.0% (0/{total_episodes})", end='', flush=True)

    # 获取第一个trainer的agent引用（用于全局最优模型）
    global_agent = trainers[0].agent if trainers else None

    try:
        while True:
            # 检查中断信号
            if interrupt_flag:
                print("\n保存中断状态...")
                for trainer in trainers:
                    trainer.save_interrupt_results(round_num)

                # 保存全局中断状态
                if global_agent:
                    interrupt_global_path = "a2c_fajsp_model_global.pth"
                    torch.save({
                        'global_best_policy_state': global_agent.global_best_policy_state,
                        'global_best_value_state': global_agent.global_best_value_state,
                        'global_best_avg_ct': global_agent.global_best_avg_ct,
                        'global_best_round': global_agent.global_best_round,
                        'round_avg_ct_history': round_avg_ct_list,
                        'interrupt_round': round_num,
                        'total_global_episodes': global_episode,
                        'interrupt_time': time.strftime("%Y-%m-%d %H:%M:%S")
                    }, interrupt_global_path)

                print("中断状态已保存，程序退出。")
                break

            # 检查文件中断
            if os.path.exists("STOP_a2c.txt"):
                print("\n检测到STOP_a2c.txt文件，保存状态...")
                for trainer in trainers:
                    trainer.save_interrupt_results(round_num)

                # 保存全局中断状态
                if global_agent:
                    interrupt_global_path = "a2c_fajsp_model_global.pth"
                    torch.save({
                        'global_best_policy_state': global_agent.global_best_policy_state,
                        'global_best_value_state': global_agent.global_best_value_state,
                        'global_best_avg_ct': global_agent.global_best_avg_ct,
                        'global_best_round': global_agent.global_best_round,
                        'round_avg_ct_history': round_avg_ct_list,
                        'interrupt_round': round_num,
                        'total_global_episodes': global_episode,
                        'interrupt_time': time.strftime("%Y-%m-%d %H:%M:%S")
                    }, interrupt_global_path)

                os.remove("STOP_a2c.txt")
                break

            all_done = all(t.episode_count >= total_episodes_per_instance for t in trainers)
            if all_done:
                break

            # 每个轮次开始前重置计数器
            round_total_ct = 0
            round_instance_count = 0

            # 轮流训练：每个算例训练1个周期
            for trainer in trainers:
                if trainer.episode_count >= total_episodes_per_instance:
                    continue

                total_reward, current_ct, pol_loss, val_loss = trainer.train_one_episode()

                round_total_ct += current_ct
                round_instance_count += 1
                global_episode += 1

            # 计算本轮次的平均完工时间
            if round_instance_count > 0:
                round_avg_ct = round_total_ct / round_instance_count
                round_avg_ct_list.append(round_avg_ct)

                # 更新平均CT Visdom曲线
                viz.line(
                    X=np.array([round_num]),
                    Y=np.array([round_avg_ct]),
                    win=avg_ct_win,
                    update='append',
                    opts=dict(title='Round Average Completion Time (A2C)', xlabel='Round', ylabel='Average CT')
                )

                # 更新进度条
                progress = min(100, global_episode / total_episodes * 100)
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r全局进度: {bar} {progress:.1f}% ({global_episode}/{total_episodes})", end='', flush=True)

                round_num += 1

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")

    # 保存最终结果
    if not interrupt_flag:
        for trainer in trainers:
            save_path = trainer.save_results()
            print(f"✅ {trainer.config_str}: 最优完工时间: {trainer.agent.best_completion_time:.2f}")

        # 保存全局最优模型
        if global_agent and global_agent.global_best_policy_state is not None:
            global_best_path = "a2c_fajsp_global_best.pth"
            torch.save({
                'global_best_policy_state': global_agent.global_best_policy_state,
                'global_best_value_state': global_agent.global_best_value_state,
                'global_best_avg_ct': global_agent.global_best_avg_ct,
                'global_best_round': global_agent.global_best_round,
                'round_avg_ct_history': round_avg_ct_list
            }, global_best_path)
            print(f"\n🎯 全局最优模型已保存: {global_best_path}")
            print(
                f"   最优平均完工时间: {global_agent.global_best_avg_ct:.2f} (第 {global_agent.global_best_round} 轮)")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总训练时间: {total_time:.1f}秒")
    print("各配置最优完工时间:")
    for trainer in trainers:
        print(f"  {trainer.config_str}: {trainer.agent.best_completion_time:.2f}")
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

    # 测试用的小列表
    test_list = [[4, 2, 2, 4, 2, 2], [4, 2, 2, 4, 2, 4], [4, 2, 2, 4, 2, 6]]

    print("A2C算法轮训训练开始")
    print("按 Ctrl+F2 中断训练")

    # 开始轮训训练
    trainers = train_round_robin(data_list, total_episodes_per_instance=10000, n_steps=10)