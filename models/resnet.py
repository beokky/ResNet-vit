import torch
import torch.nn as nn


# 用于浅层网络，适用于resnet18 34的残差结构
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中主分支的卷积核个数有没有发生变化

    # 初始化函数，定义网络层和一些参数
    def __init__(self, in_channel, out_channel, stride = 1, downsample=None):
        super(BasicBlock, self).__init__()
        # 传入输入通道数， 输出通道数， 卷积核大小默认为3
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # downsample对应虚线残差结构捷径中的1×1卷积

    def forward(self, x):
        identity = x  # 保存输入数据，便于后面进行残差链接
        if self.downsample is not None:  # 如果下采样层不为空，则对输入进行下采样得到捷径分支的输出
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 将输出与残差连接相加
        out = self.relu(out)

        return out


# 用于深层网络,resnet50 101
class Bottleneck(nn.Module):
    expansion = 4  # 指定扩张因子，主分支的卷积核个数最后一层变成第一层四倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 定义第一个1*1卷积核，压缩通道数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, padding=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample   # 如果输入和输出的尺寸不匹配，则对它进行下采样

    def forward(self,x):
        identity = x  # 保存输入数据，便于后面进行残差链接
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)  # 第一个卷积核改变通道数大小，通道数压缩
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 大小#*3
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)  # 恢复通道数
        out = self.bn3(out)
        out += identity  # 将主分支与残差分支相加
        out = self.relu(out)

        return out

# 定义网络框架部分
class ResNet(nn.Module):
    # block为对应网络选取 比如resnet18 34选取的block为basicblock ，resnet50 101选取的为bottleneck
    # blocks_num 残差结构的数目， 比如resnet18 为[2,2,2,2]
    # num_classes 分类数为1000,表示模型的输出类别数为 1000
    # include_top分类头 为线性层
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super().__init__()
        self.include_top = include_top  # 分类头
        self.in_channel = 64
        # 定义第一个卷积层，输入通道数为3 RGB三通道故为3 若为灰度图 输入通道数需要改为1 ，将图像大小减半
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, bias=False, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # blocks_num在resnet18 内是 [2,2,2,2]
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 创建四个残差层，分别对应resnet的四个stage
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # 如果 include_top=True，则添加自适应平均池化层和全连接层，用于分类任务。
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 经过自适应平均池化下采样，输出的大小为1*1
            self.fc = nn.Linear(512*block.expansion,num_classes) # 输入的节点个数为自适应平均池化后的节点个数，因为高宽为1，所以节点个数为channel的数量
        # 初始化卷积层的权重
        for m in self.modules():  # 如果是卷积层
            if isinstance(m,nn.Conv2d):
                # 使用kaiming初始化
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")


    def _make_layer(self, block, channel, block_num, stride=1):
        # block为对应网络深度来选取
        # channel 为残差结构中第一个卷积层的个数
        # block_num 该层包含多少个残差结构
        downsample = None
        # 如果步长不为1或者输入通道数不等于残差块的输入通道数*扩张因子，则需要进行下采样
        # 下采样是为了保证输入通道数和输出通道数对齐，以进行残差操作
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 对于layer1的构建 使用resnet18 不满足条件 会跳过下采样的操作
            # 对于resnet50 101  满足条件 会进行下采样操作 通道数由64 变为256 需要对齐便于进行残差连接
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []  # 定义一个空列表 用于存储残差结构
        # block为我们选取的basickblock和bottleneck，self.in_channel输入特征图的通道数 64
        # channel 残差快对应主分支的第一个卷积核的个数
        # downsample 下采样的操作
        layers.append(block(
            self.in_channel,
            channel,
            downsample=downsample,
            stride=stride
        ))
        self.in_channel = channel * block.expansion  # 特征图已经经过了一次残差结构，18 34来说in_channel 不会改变，resnet50 101 inchannel会变为4倍
        # 通过循环将一系列实线的残差结构写入进去，无论18 34 50 101 从第二层开始都是实线的残差结构
        # 传入输入特征图的通道数和残差结构主分支上第一层卷积的卷积核个数
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)  # 构建layers后通过nn.Sequential将一系列的残差结构组合在一起，得到layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 通过3*3的最大池化
        # 将输出输入到layer1 即conv2对应的一系列残差结构
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x  # 返回输出的结果


def resnet18(num_classes=1000,include_top=True,pretrained=False):
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes,include_top=include_top)

def resnet34(num_classes=1000,include_top=True,pretrained=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)

def resnet50(num_classes=1000,include_top=True,pretrained=False):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck,[3,4,6,3],num_classes=num_classes,include_top=include_top)

def resnet101(num_classes=1000,include_top=True,pretrained=False):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck,[3,4,23,3],num_classes=num_classes,include_top=include_top)

