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

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


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


# Actor-Critic Agent with GAE
class ACAgent:
    def __init__(self, action_feature_dim, combined_dim, mlp_value_dim,
                 policy_lr=1e-4, value_lr=1e-4, gamma=0.99, gae_lambda=0.95,
                 batch_size=32):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        self.policy_net = PolicyNetwork(action_feature_dim, combined_dim).to(self.device)
        self.value_net = ValueNetwork(mlp_value_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)

        self.transitions = []

    def take_action(self, matrix_states, candidate_actions, candidate_actions_features):
        """选择动作 - 返回动作索引"""
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
        """更新网络 - 修复版本"""
        # 只有在经验达到一定数量时才更新
        if len(self.transitions) < self.batch_size:
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

        # 2. 计算当前状态价值（需要梯度）
        state_values = []
        for s in states:
            value = self.value_net(s)
            # 确保是标量
            if value.dim() > 0:
                value = value.squeeze()
            state_values.append(value)
        state_values = torch.stack(state_values)

        # 3. 计算下一个状态价值（不需要梯度）
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
            # 重新计算动作概率（保持梯度）
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

        # 8. 计算价值损失 - 使用 Huber Loss
        critic_loss = F.smooth_l1_loss(state_values, returns.detach())

        # 调试信息
        if len(self.transitions) % 100 == 0:
            print(f"Advantages: [{advantages.min().item():.3f}, {advantages.max().item():.3f}]")
            print(f"Returns: [{returns.min().item():.3f}, {returns.max().item():.3f}]")

        # 9. 更新策略网络
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # 10. 更新价值网络
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        # 清空经验
        self.transitions.clear()

        return actor_loss.item(), critic_loss.item()


class ACTrainer:
    def __init__(self, env, agent, max_episodes=None):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes

        self.episode_rewards = []
        self.completion_times = []
        self.policy_losses = []
        self.value_losses = []

    def train(self):
        print("开始训练...")
        start = time.time()

        for episode in range(self.max_episodes):
            self.env.reset()
            state = self.env.current_state
            total_reward = 0
            steps = 0

            while True:
                candidate_actions = self.env.get_candidate_actions()
                if not candidate_actions:
                    break

                candidate_actions_features = {
                    a: self.env.action_features(a)[a]
                    for a in candidate_actions
                }

                # 选择动作
                action_idx, state_value = self.agent.take_action(
                    state, candidate_actions, candidate_actions_features
                )
                action = candidate_actions[action_idx]

                # 执行动作
                next_state, reward, done = self.env.step(action)



                total_reward += reward
                steps += 1

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

            # 更新网络
            actor_loss, critic_loss = self.agent.update()

            # 记录数据
            self.episode_rewards.append(total_reward)
            self.completion_times.append(self.env.completion_time)
            self.policy_losses.append(actor_loss)
            self.value_losses.append(critic_loss)

            # 输出信息
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_ct = np.mean(self.completion_times[-10:])
                print(f"Episode {episode + 1}/{self.max_episodes}, "
                      f"Avg Reward={avg_reward:.3f}, "
                      f"Avg CT={avg_ct:.2f}, "
                      f"Steps={steps}, "
                      f"Actor Loss={actor_loss:.6f}, Critic Loss={critic_loss:.6f}")

        print(f"训练完成，总耗时：{time.time() - start:.2f}秒")
        return self.episode_rewards, self.completion_times


def main(M_p=None, M_a=None, product_count=None, kind_count=None, J_r=None, N_p=None, episodes=None):
    env = FAJSP_Environment(M_p=M_p, M_a=M_a, kind_count=kind_count,
                            product_count=product_count, J_r=J_r, N_p=N_p)
    env.reset()

    # 获取动作特征维度
    ca = env.get_candidate_actions()
    caf = env.action_features(ca[0])
    feature_dim = len(caf[ca[0]])

    # 获取 CNN 状态维度
    cnn = CNNFeatureExtractor(input_channels=1).to(device)
    s = env.current_state
    feats = []
    for m in s:
        t = torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)
        feats.append(cnn(t))
    sf = torch.cat(feats, dim=1)
    state_dim = sf.shape[1]

    # 创建带GAE的Agent
    ac = ACAgent(
        action_feature_dim=feature_dim,
        combined_dim=state_dim + feature_dim,
        mlp_value_dim=state_dim,
        gamma=0.99,
        gae_lambda=0.95,
        policy_lr=1e-4,  # 降低学习率
        value_lr=1e-4,  # 降低学习率
        batch_size=32  # 批量更新
    )

    trainer = ACTrainer(env, ac, max_episodes=episodes)
    trainer.train()

    # 保存模型
    torch.save({
        'policy': ac.policy_net.state_dict(),
        'value': ac.value_net.state_dict(),
        'reward': trainer.episode_rewards,
        'completion_times': trainer.completion_times,
        'policy_losses': trainer.policy_losses,
        'value_losses': trainer.value_losses
    }, f"ac_gae_{M_p}_{M_a}_{product_count}_{kind_count}_{J_r}_{N_p}_fajsp_model.pth")

    print(f"模型已保存 ac_gae_{M_p}_{M_a}_{product_count}_{kind_count}_{J_r}_{N_p}_fajsp_model.pth")


if __name__ == "__main__":
    data_list = [[4, 2, 2, 4, 2, 2]]
    data = data_list[0]  # 选择一个配置进行训练
    main(M_p=data[0], M_a=data[1], product_count=data[2], kind_count=data[3], J_r=data[4], N_p=data[5],
         episodes=1000)