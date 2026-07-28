# -*- coding: utf-8 -*-
"""
预训练策略网络的加载与贪婪评估。

models/下的checkpoint(论文C10-C18九个中等规模算例训练所得)使用的结构为:
  CNN: Conv2d(1,64,k=2,p=1) -> ReLU -> Conv2d(64,32,k=2,p=1) -> ReLU -> AdaptiveAvgPool2d(2,2)
  ATN: 单头自注意力, 特征维178
  MLP: Linear(690,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)
与论文的[64,32]CNN卷积核配置和[64,32]MLP隐层配置一致(agent/目录中的当前
组件是精简版, 结构不同), 因此本模块按checkpoint的state_dict形状动态重建网络,
不依赖agent/CNN.py等文件的当前实现。
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'agent'))

from ATN import SelfAttention  # noqa: E402  (注意力结构与checkpoint一致, 可直接复用)


def _build_cnn_from_state(state, prefix='cnn.conv_layers.'):
    """按state_dict中conv层的形状重建CNN(Sequential索引与checkpoint对齐)"""
    conv_keys = sorted(
        {int(k[len(prefix):].split('.')[0]) for k in state if k.startswith(prefix)})
    layers = []
    idx = 0
    for ck in conv_keys:
        w = state[f'{prefix}{ck}.weight']
        out_c, in_c, kh, _ = w.shape
        while idx < ck:  # 占位保持Sequential索引一致
            layers.append(nn.ReLU())
            idx += 1
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kh, padding=1))
        idx += 1
    layers.append(nn.ReLU())
    layers.append(nn.AdaptiveAvgPool2d((2, 2)))
    return nn.Sequential(*layers), conv_keys


class PretrainedCNN(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.conv_layers, _ = _build_cnn_from_state(state_dict)

    def forward(self, x):
        feat = self.conv_layers(x)
        return feat.view(feat.size(0), -1)


class PretrainedMLP(nn.Module):
    def __init__(self, state_dict, prefix='mlp.network.'):
        super().__init__()
        lin_keys = sorted(
            {int(k[len(prefix):].split('.')[0]) for k in state_dict if k.startswith(prefix)})
        layers = []
        idx = 0
        for i, lk in enumerate(lin_keys):
            w = state_dict[f'{prefix}{lk}.weight']
            while idx < lk:
                layers.append(nn.ReLU())
                idx += 1
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            idx += 1
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PretrainedPolicy(nn.Module):
    """与checkpoint结构对齐的策略网络: CNN状态特征 + ATN动作交互 + MLP打分"""

    def __init__(self, state_dict):
        super().__init__()
        atn_dim = state_dict['atn.query.weight'].shape[1]
        self.cnn = PretrainedCNN(state_dict)
        self.atn = SelfAttention(feature_dim=atn_dim, num_heads=1)
        self.mlp = PretrainedMLP(state_dict)
        self.load_state_dict(state_dict)
        self.eval()

    @torch.no_grad()
    def forward(self, matrix_states, candidate_actions, candidate_actions_features):
        feats = []
        for matrix in matrix_states:
            m = torch.tensor(matrix).unsqueeze(0).unsqueeze(0).float()
            feats.append(self.cnn(m))
        state_feature = torch.cat(feats, dim=1)

        action_feature_list = [candidate_actions_features[a] for a in candidate_actions]
        action_tensor = torch.tensor(action_feature_list).unsqueeze(0).float()
        action_trans = self.atn(action_tensor)

        k = len(candidate_actions)
        combined = torch.cat(
            [state_feature.expand(k, -1), action_trans.squeeze(0)], dim=1)
        scores = self.mlp(combined).squeeze(-1)
        return F.softmax(scores, dim=0)


def load_policy(pth_path):
    """加载checkpoint, 返回(策略网络, 元信息dict)"""
    data = torch.load(pth_path, map_location='cpu', weights_only=False)
    state = data['policy'] if data.get('policy') is not None else data.get('best_policy_state')
    policy = PretrainedPolicy(state)
    meta = {'config': data.get('config'),
            'best_completion_time': data.get('best_completion_time')}
    return policy, meta


@torch.no_grad()
def greedy_rollout(policy, env):
    """贪婪(argmax)调度一个episode, 返回(makespan, 平均每步决策时间s)"""
    env.reset()
    decision_times = []
    while True:
        candidates = env.get_candidate_actions()
        if not candidates:
            break
        t0 = time.perf_counter()
        feats = {a: env.action_features(a)[a] for a in candidates}
        probs = policy(env.current_state, candidates, feats)
        action = candidates[int(torch.argmax(probs))]
        decision_times.append(time.perf_counter() - t0)
        env.step(action)
        if env.done:
            break
    adt = sum(decision_times) / len(decision_times) if decision_times else 0.0
    return float(env.completion_time), adt
