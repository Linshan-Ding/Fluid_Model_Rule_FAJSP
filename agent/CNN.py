from torch import nn
import torch


"""
优化卷积核的权重参数
"""
class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器 - 用于提取状态矩阵的空间特征"""
    def __init__(self, input_channels=1):
        super(CNNFeatureExtractor, self).__init__()

        # 卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=2,padding=1),
            nn.ReLU(),
            # nn.Conv2d(64, 32, kernel_size=2, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # 全局池化
            nn.AdaptiveAvgPool2d((2, 2))  # 统一到 2×2
        )


    def forward(self, x):
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        return conv_features

if __name__ == "__main__":
    # 最简单的测试
    model = CNNFeatureExtractor()
    x = torch.randn(1, 1, 16, 16)  # 1个样本，1通道，16x16大小
    output = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")