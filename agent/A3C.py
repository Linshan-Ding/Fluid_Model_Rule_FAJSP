# 异步多线程A3C算法（优势演员-评论家算法）实现, 包含GAE优势估计 - 轮训模式
# 目前版本是所有配置共享一个全局网络参数，固定维度初始化：使用第一个配置的维度初始化网络，所以需要再env里统一所有状态矩阵的维度
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
import threading
from collections import deque
import queue
import signal
import visdom

# 全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化Visdom（只用于显示第一线程）
viz = visdom.Visdom(env='FAJSP_A3C_Training')

# 全局中断标志
interrupt_flag = False


def signal_handler(sig, frame):
    """处理中断信号（Ctrl+F2）"""
    global interrupt_flag
    interrupt_flag = True
    print("\n检测到中断信号 (Ctrl+F2)，正在保存当前状态...")


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)


# ====================== 全局网络 ======================
class GlobalNetworks:
    def __init__(self, action_feature_dim, combined_dim, mlp_value_dim):
        self.device = device

        # 全局策略网络
        self.policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.policy_net.share_memory()  # 允许多线程共享

        # 全局价值网络
        self.value_net = ValueNetwork(mlp_value_dim).to(self.device)
        self.value_net.share_memory()

        # 全局优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-4)

        # 全局计数器
        self.global_step = 0
        self.lock = threading.Lock()  # 用于更新参数时加锁

        # 全局最优记录（基于第一线程平均CT）
        self.global_best_policy_state = None
        self.global_best_value_state = None
        self.global_best_avg_ct = float('inf')
        self.global_best_round = -1


# ====================== 策略网络和价值网络 ======================
class PolicyNetwork(nn.Module):
    def __init__(self, action_feature_dim, combined_dim):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.cnn = CNNFeatureExtractor(input_channels=1).to(self.device)
        self.atn = SelfAttention(feature_dim=action_feature_dim, num_heads=1).to(self.device)
        self.mlp = MLP(input_dim=combined_dim).to(self.device)

    def forward(self, matrix_states, candidate_actions, candidate_actions_features):
        state_features = []
        for matrix in matrix_states:
            m = torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float().to(self.device)
            feat = self.cnn(m)
            state_features.append(feat)
        state_feature = torch.cat(state_features, dim=1)

        action_feature_list = []
        for a in candidate_actions:
            action_feature_list.append(candidate_actions_features[a])
        action_tensor = torch.tensor(action_feature_list).unsqueeze(0).float().to(self.device)
        action_trans = self.atn(action_tensor)

        scores = []
        for i in range(len(candidate_actions)):
            af = action_trans[:, i, :]
            combined = torch.cat([state_feature, af], dim=1)
            score = self.mlp(combined)
            scores.append(score)
        scores_tensor = torch.cat(scores, dim=1).squeeze(0)
        action_probs = F.softmax(scores_tensor, dim=0)
        return action_probs


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


# ====================== 异步更新线程 ======================
class AsyncUpdateThread(threading.Thread):
    """专用更新线程：从队列获取梯度并更新全局网络"""

    def __init__(self, global_nets, policy_grad_queue, value_grad_queue):
        super().__init__()
        self.global_nets = global_nets
        self.policy_grad_queue = policy_grad_queue
        self.value_grad_queue = value_grad_queue
        self.update_count = 0
        self.daemon = True  # 设为守护线程

    def run(self):
        print("异步更新线程启动")
        while True:
            try:
                # 从队列获取梯度（阻塞等待）
                policy_grads = self.policy_grad_queue.get(timeout=1.0)
                value_grads = self.value_grad_queue.get(timeout=1.0)

                # 更新全局网络
                self.update_global_networks(policy_grads, value_grads)
                self.update_count += 1

            except queue.Empty:
                # 队列为空，检查是否所有工作线程都已完成
                continue
            except Exception as e:
                print(f"异步更新线程错误: {e}")
                break

        print("异步更新线程结束")

    def update_global_networks(self, policy_grads, value_grads):
        """使用梯度更新全局网络"""
        with self.global_nets.lock:
            # 更新策略网络
            self.global_nets.policy_optimizer.zero_grad()
            for param, grad in zip(self.global_nets.policy_net.parameters(), policy_grads):
                if grad is not None:
                    param.grad = grad
            self.global_nets.policy_optimizer.step()

            # 更新价值网络
            self.global_nets.value_optimizer.zero_grad()
            for param, grad in zip(self.global_nets.value_net.parameters(), value_grads):
                if grad is not None:
                    param.grad = grad
            self.global_nets.value_optimizer.step()

            self.global_nets.global_step += 1


# ====================== A3C 工作者线程（轮训模式） ======================
class A3CWorker(threading.Thread):
    def __init__(self, worker_id, global_nets, data_configs,
                 policy_grad_queue, value_grad_queue,
                 gamma=0.99, gae_lambda=0.95, max_steps=5,
                 episodes_per_config=1000, viz_enabled=True):
        super().__init__()
        self.worker_id = worker_id
        self.global_nets = global_nets
        self.device = global_nets.device
        self.data_configs = data_configs

        # 本地网络（复制全局网络）
        self.action_feature_dim = None
        self.combined_dim = None
        self.mlp_value_dim = None
        self.policy_net = None
        self.value_net = None

        # 环境和参数
        self.env = None
        self.current_config = None
        self.config_str = None
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_steps = max_steps
        self.episodes_per_config = episodes_per_config
        self.viz_enabled = viz_enabled and worker_id == 0  # 只有第一线程显示Visdom

        # 梯度队列
        self.policy_grad_queue = policy_grad_queue
        self.value_grad_queue = value_grad_queue

        # 本地经验缓存
        self.transitions = []

        # 线程私有记录
        self.episode_rewards = []  # 所有配置的奖励
        self.completion_times = []  # 所有配置的完工时间
        self.config_records = {}  # 每个配置的记录
        self.config_best_ct = {}  # 每个配置的最优CT
        self.current_episode_in_config = 0  # 当前配置的训练周期数

        # Visdom窗口（仅第一线程）- 为每个配置创建窗口
        if self.viz_enabled:
            self.config_windows = {}  # 存储每个配置的Visdom窗口
            self.config_ct_data = {}  # 存储每个配置的CT数据

        # 最优记录（只用于第一线程）
        if worker_id == 0:
            self.best_config_info = None
            self.best_ct = float('inf')
            self.best_episode = -1
            # 为每个配置创建最优记录
            self.config_best_policy_state = {}
            self.config_best_value_state = {}
            self.config_best_completion_time = {}

    def init_networks(self, env_params):
        """根据环境参数初始化网络维度"""
        # 创建测试环境以获取维度
        test_env = FAJSP_Environment(**env_params)
        test_env.reset()

        ca = test_env.get_candidate_actions()
        if not ca:
            raise ValueError(f"配置 {self.config_str} 没有候选动作")

        caf = test_env.action_features(ca[0])
        feature_dim = len(caf[ca[0]])

        cnn = CNNFeatureExtractor(input_channels=1).to(self.device)
        s = test_env.current_state
        feats = []
        for m in s:
            t = torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(self.device)
            feats.append(cnn(t))
        sf = torch.cat(feats, dim=1)
        state_dim = sf.shape[1]

        # 更新维度
        self.action_feature_dim = feature_dim
        self.combined_dim = state_dim + feature_dim
        self.mlp_value_dim = state_dim

        # 创建本地网络
        self.policy_net = PolicyNetwork(feature_dim, state_dim + feature_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        self.sync_with_global()

    def sync_with_global(self):
        """从全局网络同步参数到本地网络"""
        if self.policy_net is None or self.value_net is None:
            return

        with self.global_nets.lock:
            self.policy_net.load_state_dict(self.global_nets.policy_net.state_dict())
            self.value_net.load_state_dict(self.global_nets.value_net.state_dict())

    def setup_environment(self, config):
        """设置当前训练的环境配置"""
        self.current_config = config
        self.config_str = f"{config[0]}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{config[5]}"

        # 创建环境参数
        env_params = {
            'M_p': config[0], 'M_a': config[1],
            'product_count': config[2], 'kind_count': config[3],
            'J_r': config[4], 'N_p': config[5]
        }

        # 初始化网络（如果需要）
        if self.action_feature_dim is None:
            self.init_networks(env_params)

        # 创建环境
        self.env = FAJSP_Environment(**env_params)
        self.env.reset()

        # 初始化配置记录
        if self.config_str not in self.config_records:
            self.config_records[self.config_str] = {
                'episodes': [],
                'rewards': [],
                'ct_values': [],
                'best_ct': float('inf')
            }

        # 为当前配置创建Visdom窗口（仅第一线程）
        if self.viz_enabled and self.config_str not in self.config_windows:
            self.config_windows[self.config_str] = viz.line(
                X=np.array([0]),
                Y=np.array([0]),
                opts=dict(
                    title=f'A3C Worker {self.worker_id} - {self.config_str}',
                    xlabel='Episode',
                    ylabel='Completion Time',
                    env='FAJSP_A3C_Training'
                )
            )
            self.config_ct_data[self.config_str] = {'x': [], 'y': []}

    def run(self):
        """轮训训练所有配置 - 真正的轮训：每个配置训练1个周期"""
        global interrupt_flag

        print(f"Worker {self.worker_id} 启动，共有 {len(self.data_configs)} 个配置需要训练")

        # 记录开始时间
        start_time = time.time()

        # 计算总周期数
        total_episodes = len(self.data_configs) * self.episodes_per_config
        completed_episodes = 0

        # 轮训训练
        while completed_episodes < total_episodes:
            # 检查中断
            if interrupt_flag or os.path.exists("STOP_a3c.txt"):
                print(f"\nWorker {self.worker_id}: 检测到中断，停止训练")
                interrupt_flag = True
                break

            # 轮训遍历所有配置
            for config in self.data_configs:
                # 检查中断
                if interrupt_flag or os.path.exists("STOP_a3c.txt"):
                    break

                # 计算当前配置的周期序号
                current_episode_in_config = completed_episodes // len(self.data_configs)

                # 如果这个配置已经完成了所有周期，跳过
                if current_episode_in_config >= self.episodes_per_config:
                    continue

                # 设置当前配置的环境
                self.setup_environment(config)

                # 训练当前配置的1个周期
                self.train_one_episode_in_config(config, current_episode_in_config)

                completed_episodes += 1

                # 检查是否完成所有周期
                if completed_episodes >= total_episodes:
                    break

        # 计算训练时间
        training_time = time.time() - start_time
        print(f"Worker {self.worker_id} 完成，总训练时间: {training_time:.2f}秒")

        # 保存工作者结果（仅第一线程）
        if self.worker_id == 0:
            self.save_all_config_results()

    def train_one_episode_in_config(self, config, episode_idx):
        """训练单个配置的单个周期"""
        global interrupt_flag

        # 同步参数
        self.sync_with_global()

        # 重置环境开始新的一轮
        self.env.reset()
        state = self.env.current_state
        total_reward = 0
        finish = False

        while not finish:
            # n步收集经验
            for step in range(self.max_steps):
                candidate_actions = self.env.get_candidate_actions()
                if not candidate_actions:
                    break

                candidate_actions_features = {
                    a: self.env.action_features(a)[a] for a in candidate_actions
                }

                # 选择动作
                with torch.no_grad():
                    action_probs = self.policy_net(state, candidate_actions, candidate_actions_features)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action_idx = action_dist.sample().item()
                    action = candidate_actions[action_idx]

                    state_value = self.value_net(state)
                    if state_value.dim() > 0:
                        state_value = state_value.squeeze()

                # 执行动作
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # 存储经验
                self.transitions.append({
                    'states': state,
                    'action_idx': action_idx,
                    'state_value': state_value.item(),
                    'rewards': reward,
                    'next_states': next_state,
                    'dones': done,
                    'candidate_actions': candidate_actions,
                    'candidate_actions_features': candidate_actions_features
                })

                state = next_state
                if done:
                    finish = True
                    break

            # 计算梯度并放入队列（异步）
            if len(self.transitions) > 0:
                self.compute_and_send_gradients()

        # 记录结果
        current_ct = self.env.completion_time
        self.episode_rewards.append(total_reward)
        self.completion_times.append(current_ct)

        # 更新配置记录
        if self.config_str not in self.config_records:
            self.config_records[self.config_str] = {
                'episodes': [],
                'rewards': [],
                'ct_values': [],
                'best_ct': float('inf')
            }

        self.config_records[self.config_str]['episodes'].append(episode_idx)
        self.config_records[self.config_str]['rewards'].append(total_reward)
        self.config_records[self.config_str]['ct_values'].append(current_ct)

        # 更新最优记录
        if current_ct < self.config_records[self.config_str]['best_ct']:
            self.config_records[self.config_str]['best_ct'] = current_ct
            self.config_best_ct[self.config_str] = current_ct

        # 第一线程的特殊处理
        if self.worker_id == 0:
            # 更新第一线程全局最优记录
            if current_ct < self.best_ct:
                self.best_ct = current_ct
                self.best_config_info = {
                    'config': config,
                    'config_str': self.config_str,
                    'ct': current_ct,
                    'episode': episode_idx,
                    'schedule_log': copy.deepcopy(getattr(self.env, 'schedule_log', []))
                }
                self.best_episode = episode_idx

            # 更新配置最优参数（像AC代码那样）
            if self.config_str not in self.config_best_completion_time or current_ct < self.config_best_completion_time[
                self.config_str]:
                self.config_best_completion_time[self.config_str] = current_ct
                self.config_best_policy_state[self.config_str] = copy.deepcopy(self.policy_net.state_dict())
                self.config_best_value_state[self.config_str] = copy.deepcopy(self.value_net.state_dict())

            # 更新Visdom（每个episode都显示）
            if self.viz_enabled and self.config_str in self.config_ct_data:
                # 收集数据点
                self.config_ct_data[self.config_str]['x'].append(episode_idx + 1)  # 从1开始
                self.config_ct_data[self.config_str]['y'].append(current_ct)

                # 更新Visdom显示
                viz.line(
                    X=np.array(self.config_ct_data[self.config_str]['x']),
                    Y=np.array(self.config_ct_data[self.config_str]['y']),
                    win=self.config_windows[self.config_str],
                    update='replace',
                    opts=dict(
                        title=f'A3C Worker {self.worker_id} - {self.config_str}',
                        xlabel='Episode',
                        ylabel='Completion Time'
                    )
                )

    def compute_gae_advantages(self, rewards, values, next_values, dones):
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

    def compute_and_send_gradients(self):
        """计算梯度并发送到队列"""
        # 准备数据
        states = [t['states'] for t in self.transitions]
        action_idxs = [t['action_idx'] for t in self.transitions]
        rewards = torch.tensor([t['rewards'] for t in self.transitions], dtype=torch.float32).to(self.device)
        next_states = [t['next_states'] for t in self.transitions]
        dones = torch.tensor([t['dones'] for t in self.transitions], dtype=torch.float32).to(self.device)
        candidate_actions_list = [t['candidate_actions'] for t in self.transitions]
        candidate_actions_features_list = [t['candidate_actions_features'] for t in self.transitions]

        # 计算价值
        state_values = []
        for s in states:
            value = self.value_net(s)
            if value.dim() > 0:
                value = value.squeeze()
            state_values.append(value)
        state_values = torch.stack(state_values)

        next_state_values = []
        for ns in next_states:
            with torch.no_grad():
                next_value = self.value_net(ns)
                if next_value.dim() > 0:
                    next_value = next_value.squeeze()
                next_state_values.append(next_value)
        next_state_values = torch.stack(next_state_values)

        # 计算GAE优势
        advantages = self.compute_gae_advantages(
            rewards, state_values.detach(), next_state_values, dones
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + state_values.detach()

        # 计算log_probs
        log_probs = []
        for i in range(len(states)):
            action_probs = self.policy_net(
                states[i], candidate_actions_list[i], candidate_actions_features_list[i]
            )
            log_prob = torch.log(action_probs[action_idxs[i]] + 1e-10)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)

        # 计算损失
        actor_loss = -torch.mean(log_probs * advantages)
        critic_loss = F.mse_loss(state_values, returns.detach())
        total_loss = actor_loss + critic_loss

        # 计算梯度
        self.policy_net.zero_grad()
        self.value_net.zero_grad()
        total_loss.backward()

        # 获取梯度
        policy_grads = [param.grad.clone() if param.grad is not None else None
                        for param in self.policy_net.parameters()]
        value_grads = [param.grad.clone() if param.grad is not None else None
                       for param in self.value_net.parameters()]

        # 将梯度放入队列（异步操作）
        self.policy_grad_queue.put(policy_grads)
        self.value_grad_queue.put(value_grads)

        # 清空本地经验
        self.transitions.clear()

    def save_all_config_results(self):
        """保存所有配置的结果"""
        for config in self.data_configs:
            config_str = f"{config[0]}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{config[5]}"

            # 收集甘特图信息（需要重新创建环境来获取）
            temp_env = FAJSP_Environment(
                M_p=config[0], M_a=config[1],
                product_count=config[2], kind_count=config[3],
                J_r=config[4], N_p=config[5]
            )

            save_path = f"a3c_fajsp_model_{config_str}.pth"

            save_data = {
                'config': config,
                'policy': self.config_best_policy_state.get(config_str, None),
                'value': self.config_best_value_state.get(config_str, None),
                'reward': self.config_records.get(config_str, {}).get('rewards', []),
                'completion_times': self.config_records.get(config_str, {}).get('ct_values', []),
                'best_completion_time': self.config_records.get(config_str, {}).get('best_ct', float('inf')),
                'best_schedule_log': self.best_config_info.get('schedule_log', []) if self.best_config_info else [],
                'gantt_info': {
                    'kind_task_tuple': getattr(temp_env, 'kind_task_tuple', []),
                    'kind_task_tuple_r': getattr(temp_env, 'kind_task_tuple_r', []),
                    'kind_task_tuple_a': getattr(temp_env, 'kind_task_tuple_a', []),
                    'machine_tuple': getattr(temp_env, 'machine_tuple', [])
                },
                'total_episodes': len(self.config_records.get(config_str, {}).get('rewards', [])),
                'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'interrupted': interrupt_flag or os.path.exists("STOP_a3c.txt")  # 标记是否中断
            }

            torch.save(save_data, save_path)
            print(
                f"✅ 配置 {config_str}: 最优完工时间: {self.config_records.get(config_str, {}).get('best_ct', float('inf')):.2f}，已保存到 {save_path}")


# ====================== 主训练函数（轮训模式） ======================
def main_a3c_round_robin(data_configs, episodes_per_config=1000, n_steps=5, num_workers=4):
    global interrupt_flag

    print("A3C轮训训练开始")
    print("按 Ctrl+F2 中断训练")
    print("创建 STOP_a3c.txt 文件也可以中断训练")
    print("=" * 60)

    # 启动前清理可能存在的旧中断文件
    if os.path.exists("STOP_a3c.txt"):
        os.remove("STOP_a3c.txt")

    # 使用第一个配置初始化网络维度
    first_config = data_configs[0]
    env_params = {
        'M_p': first_config[0], 'M_a': first_config[1],
        'product_count': first_config[2], 'kind_count': first_config[3],
        'J_r': first_config[4], 'N_p': first_config[5]
    }

    # 创建测试环境以获取维度
    test_env = FAJSP_Environment(**env_params)
    test_env.reset()

    ca = test_env.get_candidate_actions()
    caf = test_env.action_features(ca[0])
    feature_dim = len(caf[ca[0]])

    cnn = CNNFeatureExtractor(input_channels=1).to(device)
    s = test_env.current_state
    feats = []
    for m in s:
        t = torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)
        feats.append(cnn(t))
    sf = torch.cat(feats, dim=1)
    state_dim = sf.shape[1]

    # 创建全局网络
    global_nets = GlobalNetworks(
        action_feature_dim=feature_dim,
        combined_dim=state_dim + feature_dim,
        mlp_value_dim=state_dim
    )

    # 创建梯度队列
    policy_grad_queue = queue.Queue(maxsize=100)  # 限制队列大小防止内存爆炸
    value_grad_queue = queue.Queue(maxsize=100)

    # 创建并启动异步更新线程
    update_thread = AsyncUpdateThread(
        global_nets=global_nets,
        policy_grad_queue=policy_grad_queue,
        value_grad_queue=value_grad_queue,
    )
    update_thread.start()

    # 创建并启动工作者线程
    workers = []
    for i in range(num_workers):
        worker = A3CWorker(
            worker_id=i,
            global_nets=global_nets,
            data_configs=data_configs,  # 所有配置列表
            policy_grad_queue=policy_grad_queue,
            value_grad_queue=value_grad_queue,
            max_steps=n_steps,
            episodes_per_config=episodes_per_config,
            viz_enabled=(i == 0)  # 只有第一线程显示Visdom
        )
        workers.append(worker)
        worker.start()

    try:
        # 等待所有工作者线程完成
        for worker in workers:
            worker.join()

        # 等待所有梯度被处理
        while not policy_grad_queue.empty() or not value_grad_queue.empty():
            time.sleep(0.1)

        # 给更新线程一点时间处理剩余梯度
        time.sleep(1)

    except KeyboardInterrupt:
        print("\n检测到键盘中断，正在保存状态...")
        interrupt_flag = True


    finally:

        print("\n" + "=" * 60)
        print("训练结束")
        print(f"总共进行了 {global_nets.global_step} 次全局更新")
        # 收集所有线程的结果
        all_config_best_ct = {}

        for worker in workers:
            for config_str, ct in worker.config_best_ct.items():
                if config_str not in all_config_best_ct or ct < all_config_best_ct[config_str]:
                    all_config_best_ct[config_str] = ct
        # 打印每个配置在所有线程的最优CT
        print("\n每个配置在所有线程的最优CT:")

        for config in data_configs:
            config_str = f"{config[0]}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{config[5]}"
            best_ct = all_config_best_ct.get(config_str, float('inf'))
            print(f"  {config_str}: {best_ct:.2f}")
        print("=" * 60)
        # 保存所有配置的结果（无论中断还是正常结束）
        if workers and workers[0].worker_id == 0:
            workers[0].save_all_config_results()

        # 保存全局最优模型

        if workers and workers[0].completion_times:
            avg_ct = np.mean(workers[0].completion_times)
            if avg_ct < global_nets.global_best_avg_ct:
                global_nets.global_best_avg_ct = avg_ct
                global_nets.global_best_policy_state = copy.deepcopy(global_nets.policy_net.state_dict())
                global_nets.global_best_value_state = copy.deepcopy(global_nets.value_net.state_dict())
                torch.save({
                    'global_best_policy_state': global_nets.global_best_policy_state,
                    'global_best_value_state': global_nets.global_best_value_state,
                    'global_best_avg_ct': global_nets.global_best_avg_ct,
                    'all_config_best_ct': all_config_best_ct,
                    'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'interrupted': interrupt_flag or os.path.exists("STOP_a3c.txt")
                }, f"a3c_global_best_model.pth")
                print(f"全局最优模型已保存为 a3c_global_best_model.pth")
                print(f"第一线程平均CT: {avg_ct:.2f}")


# ====================== 主入口 ======================
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

    print("=" * 60)
    print("A3C轮训训练参数:")
    print(f"配置数量: {len(test_list)}")
    print(f"每个配置训练周期数: {1000}")
    print(f"工作线程数: {4}")
    print(f"最大步数: {10}")
    print("=" * 60)

    # 开始轮训训练
    main_a3c_round_robin(
        data_configs=data_list,
        episodes_per_config=1000000,
        n_steps=10,
        num_workers=4
    )