# -*- coding: utf-8 -*-
"""
无可视化依赖的PPO训练核心(云端/无头环境用)。

网络结构与超参数与agent/PPO.py完全一致(CNN状态特征 + 自注意力动作特征 + MLP打分,
lr_policy=3e-4, lr_value=1e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, batch_size=64),
仅移除visdom/matplotlib依赖, 并支持向环境传入OOD算例的分布参数。

训练方式为"顺序轮询": 一个共享的策略/价值网络在D1-D12上循环训练, 每个episode
按顺序切换到下一个算例(状态矩阵统一padding到88x88、动作特征维度一致,
因此单一网络可跨算例共享参数)。episode执行逻辑见run_episode()。
"""
import os
import sys
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

from CNN import CNNFeatureExtractor  # noqa: E402
from ATN import SelfAttention  # noqa: E402
from MLP import MLP  # noqa: E402
from env.env import FAJSP_Environment  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            state_features.append(self.cnn(m))
        state_feature = torch.cat(state_features, dim=1)

        action_feature_list = [candidate_actions_features[a] for a in candidate_actions]
        action_tensor = torch.tensor(action_feature_list).unsqueeze(0).float().to(self.device)
        action_trans = self.atn(action_tensor)

        scores = []
        for i in range(len(candidate_actions)):
            af = action_trans[:, i, :]
            combined = torch.cat([state_feature, af], dim=1)
            scores.append(self.mlp(combined))
        scores_tensor = torch.cat(scores, dim=1).squeeze(0)
        return F.softmax(scores_tensor, dim=0)


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
            feats.append(self.cnn(m))
        return self.mlp(torch.cat(feats, dim=1))


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

        self.best_policy_state = None
        self.best_value_state = None
        self.best_completion_time = float('inf')

    def update_best_params(self, completion_time):
        if completion_time < self.best_completion_time:
            self.best_completion_time = completion_time
            self.best_policy_state = copy.deepcopy(self.policy_net.state_dict())
            self.best_value_state = copy.deepcopy(self.value_net.state_dict())
            return True
        return False

    def select_action(self, state, candidate_actions, candidate_actions_features):
        with torch.no_grad():
            probs = self.old_policy_net(state, candidate_actions, candidate_actions_features)
            if len(candidate_actions) == 0:
                return None, None, None, None
            idx = torch.multinomial(probs, 1).item()
            action = candidate_actions[idx]
            log_p = torch.log(probs[idx] + 1e-10)
            return action, log_p, probs[idx], candidate_actions_features[action]

    def store_transition(self, state, action, log_prob, reward, next_state, done,
                         action_features, candidate_actions, candidate_actions_features):
        self.memory.append({
            'state': state, 'action': action, 'log_prob': log_prob.clone(),
            'reward': float(reward), 'next_state': next_state, 'done': float(done),
            'action_features': action_features,
            'candidate_actions': list(candidate_actions),
            'candidate_actions_features': dict(candidate_actions_features)
        })

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        gamma, lam = self.gamma, 0.95
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * gae * (1 - dones[t])
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory]).to(self.device)
        rewards = torch.tensor([m['reward'] for m in self.memory]).float().to(self.device)
        dones = torch.tensor([m['done'] for m in self.memory]).float().to(self.device)
        states = [m['state'] for m in self.memory]
        next_states = [m['next_state'] for m in self.memory]

        with torch.no_grad():
            values = torch.tensor([self.value_net(s).item() for s in states]).to(self.device)
            next_values = torch.tensor([
                self.value_net(ns).item() if ns is not None else 0.0 for ns in next_states
            ]).to(self.device)

        advantages = self.compute_advantages(rewards, values, next_values, dones)
        returns = advantages + values
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses = [], []
        for _ in range(self.K_epochs):
            batch_indices = list(range(len(self.memory)))
            random.shuffle(batch_indices)
            for start in range(0, len(batch_indices), self.batch_size):
                idxs = batch_indices[start:start + self.batch_size]
                if len(idxs) == 0:
                    continue
                batch_old = old_log_probs[idxs]
                batch_adv = advantages[idxs]
                batch_ret = returns[idxs]

                new_log_probs, value_preds = [], []
                for mem_idx in idxs:
                    m = self.memory[mem_idx]
                    probs = self.policy_net(m['state'], m['candidate_actions'],
                                            m['candidate_actions_features'])
                    act_index = m['candidate_actions'].index(m['action'])
                    new_log_probs.append(torch.log(probs[act_index] + 1e-10))
                    value_preds.append(self.value_net(m['state']))

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


def build_shared_agent(env, **ppo_kwargs):
    """基于任一环境推断网络维度并构建共享PPO agent。

    状态矩阵统一padding到88x88, 动作特征维度对所有算例一致,
    因此由单个环境推断的维度对全部算例有效。
    """
    env.reset()
    ca = env.get_candidate_actions()
    if len(ca) == 0:
        raise ValueError("没有候选动作, 无法推断网络维度")
    caf = env.action_features(ca[0])
    feature_dim = len(caf[ca[0]])

    cnn = CNNFeatureExtractor(input_channels=1).to(device)
    feats = []
    for m in env.current_state:
        t = torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)
        feats.append(cnn(t))
    state_dim = torch.cat(feats, dim=1).shape[1]

    return PPO(action_feature_dim=feature_dim,
               combined_dim=state_dim + feature_dim,
               mlp_value_dim=state_dim, **ppo_kwargs)


def run_episode(agent, env):
    """用共享agent在指定环境上训练一个episode并执行PPO更新。

    返回 (total_reward, completion_time, policy_loss, value_loss, schedule_log)。
    """
    env.reset()
    state = env.current_state
    total_reward = 0
    while True:
        candidate_actions = env.get_candidate_actions()
        if not candidate_actions:
            break
        candidate_actions_features = {
            a: env.action_features(a)[a] for a in candidate_actions
        }
        action, logp, _, act_feat = agent.select_action(
            state, candidate_actions, candidate_actions_features)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, logp, reward, next_state, done,
                               act_feat, candidate_actions, candidate_actions_features)
        state = next_state
        total_reward += reward
        if done:
            break

    pol_loss, val_loss = agent.update()
    schedule_log = copy.deepcopy(getattr(env, "schedule_log", []))
    return total_reward, float(env.completion_time), pol_loss, val_loss, schedule_log
