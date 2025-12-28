import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            #
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

    def forward(self, x):
            return self.network(x)


# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = MLP(input_dim=128)
    print(model)

    # 测试
    test_input = torch.randn(1, 128)
    output = model(test_input)
    print(f"输出: {output}")