import torch.nn as nn
import torch
import torch.nn.functional as F


# 定义GoogLeNet模型
class GoogLeNet(nn.Module):
    # 初始参数
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)

        if self.aux_logits:
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes)
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        # 是否初始化权重
        if init_weights:
            self._initialize_weights()

    # 正向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    # 初始化权重方法
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Inception结构模板
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1, stride=1),
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1, stride=1),
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    # 正向传播过程
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 定义辅助分类器
class InceptionAux(nn.Module):
    # 参数
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    # 定义正向传播过程
    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


# 基本卷积层类【基本卷积模板】
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    # 正向传播过程
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
