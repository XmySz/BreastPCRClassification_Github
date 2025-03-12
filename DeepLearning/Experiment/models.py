"""
    Resnet18编码器+MHSA+LSTM
"""
import torch
import torch.nn as nn
import monai
from monai.networks.nets import resnet18


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, _ = self.lstm(x)
        
        # 计算注意力权重 - 在序列长度维度(dim=1)上做softmax
        attention_weights = torch.softmax(self.attention(output), dim=1)
        
        # 加权求和 - 在序列长度维度上求和
        context = torch.sum(attention_weights * output, dim=1)
        
        # 分类
        out = self.classifier(context)
        return out
    

class MultiHeadSelfAttention(torch.nn.Module):
    """多头自注意力模块"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.mhsa = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x, _ = self.mhsa(x, x, x)
        return x + residual

class ResNet18Encoder(torch.nn.Module):
    """
    基于ResNet18的3D编码器网络
    """

    def __init__(self, in_channels=1, pretrained=False, feature_size=512):
        """
        初始化3D ResNet18编码器

        参数:
            in_channels (int): 输入通道数,默认为1(灰度图像)
        """
        super().__init__()
        self.feature_size = feature_size

        # 初始化3D ResNet18
        self.encoder = monai.networks.nets.ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type="B",
            widen_factor=1.0,
            num_classes=self.feature_size,
            feed_forward=False,
            spatial_dims=3,
        )

        # self.encoder = monai.networks.nets.DenseNet121(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=512,
        #     pretrained=False
        # )

        # 移除分类头
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))

        # 添加全局平均池化层
        self.global_pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

        # 添加线性层调整特征维度
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, self.feature_size),
            torch.nn.BatchNorm1d(self.feature_size),
            torch.nn.ReLU(),
        )

        # 添加多头自注意力层
        self.mhsa = MultiHeadSelfAttention(embed_dim=self.feature_size, num_heads=4)

        # 添加LSTM层建模时间动态
        # self.lstm = torch.nn.LSTM(
        #     input_size=512,
        #     hidden_size=512,
        #     batch_first=True,
        # )

        # 添加最终的分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 2),
            torch.nn.Softmax(dim=1)
        )

        # 添加LSTM分类器
        self.lstm_classifier = LSTMClassifier(input_size=self.feature_size, hidden_size=self.feature_size, num_classes=2)

    def forward(self, x):
        """
        前向传播

        参数:
            x (tensor): 输入张量 [B, C, H, W, D]

        返回:
            tensor: 分类预测结果 [B, 2]
        """
        x = x.float()

        # 将输入按时间点分为4组
        t0, t1, t2, t3 = x.chunk(4, dim=1)  # 在通道维度上分割

        # 分别处理每个时间点
        features_list = []
        for t in [t0, t1, t2, t3]:
            # 编码
            feat = self.encoder(t)

            # 全局平均池化
            feat = self.global_pool(feat)
            feat = feat.squeeze(-1).squeeze(-1).squeeze(-1)
            # feat = feat.transpose(1, 2)

            # 线性变换
            # feat = self.fc(feat)

            # 多头自注意力
            feat = self.mhsa(feat)

            features_list.append(feat)

        # 拼接所有时间点的特征 [B, 4, 512]
        # 中间添加一个维度
        features_list = [torch.unsqueeze(feat, dim=1) for feat in features_list]
        features = torch.cat(features_list, dim=1)

        # LSTM处理时序信息
        # output, (hidden_states, cell_states) = self.lstm_classifier(features)
        output = self.lstm_classifier(features)

        # 将四个时间点的特征累加到一起并移除时间维度
        # output = output[:, -1, :]
        # output = output.sum(dim=1)

        # return self.classifier(output)
        return output


if __name__ == "__main__":
    # 创建编码器实例
    encoder = ResNet18Encoder(in_channels=1)

    # 生成随机输入数据
    batch_size = 4
    channels = 4
    height = 128
    width = 128
    depth = 32
    x = torch.randn(batch_size, channels, height, width, depth)

    # 前向传播
    features = encoder(x)

    # 打印输入输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出特征形状: {features.shape}")

    # 打印网络结构
    print("\n网络结构:")
    print(encoder)

    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n总参数量: {total_params:,}")
