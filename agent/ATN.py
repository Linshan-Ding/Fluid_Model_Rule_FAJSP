import torch
import torch.nn as nn

"""
优化四个线性变换层的权重参数
"""

class SelfAttention(nn.Module):
    """自注意力机制 - 用于理解动作特征间的关系"""

    def __init__(self, feature_dim = None, num_heads=1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim

        # 线性变换层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x shape: (batch_size, num_actions, feature_dim)
        batch_size, num_actions, feature_dim = x.shape

        # 线性变换
        Q = self.query(x).view(batch_size, num_actions, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, num_actions, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, num_actions, self.num_heads, self.head_dim)

        # 转置以便矩阵乘法
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        # 应用注意力权重
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_actions, feature_dim)

        # 最终线性变换
        out = self.fc_out(out)
        return out

if __name__ == "__main__":
    # 最简单的测试
    model = SelfAttention(feature_dim=1, num_heads=1)
    x = torch.randn(1, 10, 1)  # 1个样本，10个动作，1维特征
    output = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"输入的内容: {x}")
    print (f"输出内容: {output}")