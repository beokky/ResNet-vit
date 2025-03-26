############################################################################################################
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 随机数种子来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine annealing：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型
# 5. 新加了weight-decay权重衰减项，防止过拟合
# 6. 新加了记录每个epoch的loss和acc的log文件及可用于tensorboard可视化的文件
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir=log_path"来启动，结果通过网页 http://localhost:6006/'查看可视化结果
############################################################################################################
# --model 可选的超参如下：
# alexnet    vgg    googlenet     resnet     densenet      mobilenet     shufflenet
# efficient    convnext     vision_transformer      swin_transformer
############################################################################################################
# 训练命令示例： # python train.py --model resnet18  --batch_size 64 --lr 0.001 --epoch 100 --classes_num 4

import argparse  # 用于解析命令行参数
import torch
import torch.optim as optim  # PyTorch中的优化器
from torch.utils.data import DataLoader  # PyTorch中用于加载数据的工具
from tqdm import tqdm  # 用于在循环中显示进度条
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
import torch.nn.functional as F  # PyTorch中的函数库
from torchvision import datasets  # PyTorch中的视觉数据集
import torchvision.transforms as transforms  # PyTorch中的数据变换操作
from torch.utils.tensorboard import SummaryWriter  # 用于创建TensorBoard日志的工具
import os  # Python中的操作系统相关功能
from utils import AverageMeter, accuracy  # 自定义工具模块，用于计算模型的平均值和准确度
from models import model_dict  # 自定义模型字典，包含了各种模型的定义
import numpy as np  # NumPy库，用于数值计算
import time  # Python中的时间相关功能
import random  # Python中的随机数生成器

# 新增：用于计算 FLOPs 和 Params
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchsummary import summary

# 自定义工具类（假设已定义）
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

parser = argparse.ArgumentParser() # 导入argparse模块，用于解析命令行参数
parser.add_argument("--model_names", type=str, default="resnet18") # 添加命令行参数，指定模型名称
parser.add_argument("--model_class", type=str, default="resnet") # 指定模型类型
parser.add_argument("--pre_trained", type=bool, default=True) #指定是否使用预训练模型，默认为False
parser.add_argument("--classes_num", type=int, default=120) # 指定类别数，默认为120
parser.add_argument("--dataset", type=str, default="dataset") # 指定数据集名称，默认为"dataset"
parser.add_argument("--batch_size", type=int, default=64) #   指定批量大小，默认为64
parser.add_argument("--epoch", type=int, default=40) #  指定训练轮次数，默认为20
parser.add_argument("--lr", type=float, default=0.01) #  指定学习率，默认为0.01
parser.add_argument("--momentum", type=float, default=0.9)  # 优化器的动量，默认为 0.9
parser.add_argument("--weight-decay", type=float, default=1e-4)  # 权重衰减（正则化项），默认为 5e-4
parser.add_argument("--seed", type=int, default=33) # 指定随机种子，默认为33
parser.add_argument("--gpu-id", type=int, default=0) # 指定GPU编号，默认为0
parser.add_argument("--print_freq", type=int, default=2)  # 打印训练信息的频率，默认为 1（每个轮次打印一次）
parser.add_argument("--exp_postfix", type=str, default="logs")  # 实验结果文件夹的后缀，默认为 "logs"
parser.add_argument("--txt_name", type=str, default="train_data")  # 记录训练过程文件名

parser.add_argument("--report_path", type=str, default="report")  # 记录训练过程文件的地址
parser.add_argument("--model_pth_path", type=str, default="model_pth")  # 最佳模型参数存放地址
parser.add_argument("--runs_path", type=str, default="runs")  # logs地址

args = parser.parse_args()


# 设置随机数生成器的种子，确保实验的可重复性
def seed_torch(seed=74):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) # 设置环境变量 CUDA_VISIBLE_DEVICES，指定可见的 GPU 设备，仅在需要时使用特定的 GPU 设备进行训练

exp_name = args.exp_postfix  # 从命令行参数中获取实验名称后缀,默认为 "logs"
exp_path = "{}/{}".format(args.report_path,args.model_names)  # 创建用于记录训练过程的路径
os.makedirs(exp_path, exist_ok=True)

# dataloader
transform_train = transforms.Compose([transforms.RandomRotation(90), # 随机旋转图像
                                        transforms.Resize([256, 256]), # # 调整图像大小为 256x256 像素
                                        transforms.RandomCrop(224),  # 随机裁剪图像为 224x224 大小
                                        transforms.RandomHorizontalFlip(), # 随机水平翻转图像
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.3738, 0.3738, 0.3738),  # 对图像进行标准化
                                                            (0.3240, 0.3240, 0.3240))])
transform_test = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.3738, 0.3738, 0.3738),
                                                            (0.3240, 0.3240, 0.3240))])
trainset = datasets.ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform=transform_train)
testset = datasets.ImageFolder(root=os.path.join(args.dataset, 'val'),
                                transform=transform_test)

# 创建训练数据加载器
train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                           # 后台工作线程数量，可以并行加载数据以提高效率
                                           shuffle=True, pin_memory=True)  # 如果可用，将数据加载到 GPU 内存中以提高训练速度
# 创建测试数据加载器
test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=4,
                                          shuffle=False, pin_memory=True)

# train
def train_one_epoch(model, optimizer, train_loader):
    model.train()
    acc_recorder = AverageMeter()  # 用于记录精度的工具
    loss_recorder = AverageMeter()  # 用于记录损失的工具

    from torch.cuda.amp import GradScaler, autocast
    # 初始化 GradScaler
    scaler = GradScaler()
    # 设置梯度累积步数
    accumulation_steps = 4

    i = 0  # 初始化批次索引
    for (inputs, targets) in tqdm(train_loader, desc="train"):
        # for i, (inputs, targets) in enumerate(train_loader):
        if torch.cuda.is_available():  # 如果当前设备支持 CUDA 加速，则将输入数据和目标数据送到 GPU 上进行计算，设置 non_blocking=True 可以使数据异步加载，提高效率。

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()

        # 使用 autocast 进行混合精度计算
        with autocast():
            out = model(inputs)
            loss = F.cross_entropy(out, targets) / accumulation_steps  # 标准化损失值
            loss_recorder.update(loss.item() * accumulation_steps, n=inputs.size(0))  # 记录原始损失值
            acc = accuracy(out, targets)[0]  # 计算精度
            acc_recorder.update(acc.item(), n=inputs.size(0))  # 记录精度值

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        i += 1  # 更新批次索引

    losses = loss_recorder.avg  # 计算平均损失
    acces = acc_recorder.avg  # 计算平均精度

    return losses, acces  # 返回平均损失和平均精度

def evaluation(model, test_loader,topk=(1,)):
    # 将模型设置为评估模式，不会进行参数更新
    model.eval()
    loss_recorder = AverageMeter()
    acc_recorder = [AverageMeter() for _ in topk]

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Evaluating"):
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()

            out = model(img)
            loss = F.cross_entropy(out, label)
            loss_recorder.update(loss.item(), img.size(0))

            acc = accuracy(out, label, topk=topk)
            for i, a in enumerate(acc):
                acc_recorder[i].update(a.item(), img.size(0))

    losses = loss_recorder.avg # 计算所有批次的平均损失
    acces = [meter.avg for meter in acc_recorder]   # 计算所有批次的平均准确率
    return losses, acces # 返回平均损失和准确率

def train(model, optimizer, train_loader, test_loader, scheduler):
    since = time.time()  # 记录训练开始时间
    best_acc = -1  # 初始化最佳准确度为-1，以便跟踪最佳模型
    best_top5_acc=-1
    best_model_path = os.path.join(args.model_pth_path, args.model_names, "best.pth")

    # 新增：早停参数
    patience = 5  # 允许验证集性能不提升的轮次数
    delta = 0.001  # 认为性能提升的最小变化量
    early_stop_counter = 0  # 计数器
    early_stop = False  # 早停标志

    # 新增：计算 FLOPs 和 Params
    if torch.cuda.is_available():
        dummy_input = torch.rand(1, 3, 224, 224).cuda()
    else:
        dummy_input = torch.rand(1, 3, 224, 224)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {args.model_names}")
    print(f"Params: {total_params / 1e6:.2f}M")
    print(f"FLOPs: {total_flops / 1e9:.2f}G")
    print("--------------------------------------")


    # 写入训练过程信息
    f = open(os.path.join(exp_path, "{}.txt".format(args.txt_name)), "w")
    f.write(f"Model: {args.model_names}\n")
    f.write(f"Params: {total_params / 1e6:.2f}M\n")
    f.write(f"FLOPs: {total_flops / 1e9:.2f}G\n")
    f.write("--------------------------------------\n")

    for epoch in range(args.epoch):
        if early_stop:  # 检查是否触发早停
            print(f"Early stopping at epoch {epoch + 1}!")
            break

        print("-----------------第{}轮训练开始-------------------".format(epoch + 1))
        since_epoch = time.time()  # 记录每一轮训练开始时间
        # 在训练集上执行一个周期的训练，并获取训练损失和准确度
        train_losses, train_acces = train_one_epoch(
            model, optimizer, train_loader
        )
        # 测试集评估（新增 Top-5 Acc）
        test_losses, test_top1_acc = evaluation(model, test_loader, topk=(1, 5))
        test_top1_acc, test_top5_acc = test_top1_acc[0], test_top1_acc[1]

        # 更新test_top5_acc
        if test_top5_acc > best_top5_acc:
            best_top5_acc = test_top5_acc
        # 保存最佳模型
        if test_top1_acc > best_acc:
            best_acc = test_top1_acc
            early_stop_counter = 0  # 重置计数器
            state_dict = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'top1_acc': test_top1_acc,
                'top5_acc': test_top5_acc,
            }
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(state_dict, best_model_path)
        else:
            early_stop_counter += 1  # 未提升，计数器+1
            if early_stop_counter >= patience:
                early_stop = True  # 触发早停

        scheduler.step()  # 更新学习率调度器

        # 记录时间（ETA）
        epoch_time = time.time() - since_epoch
        remaining_time = epoch_time * (args.epoch - epoch - 1)

        # 打印信息
        msg = (
            f"训练轮次epoch= : {epoch + 1}\n"
            f"train_loss: {train_losses:.4f}, train_top1_acc: {train_acces:.2f}%\n"
            f"test_loss: {test_losses:.4f}, test_top1_acc: {test_top1_acc:.2f}%, test_top5_acc: {test_top5_acc:.2f}%\n"
            f"epoch_time: {epoch_time:.2f}s, ETA: {remaining_time:.2f}s\n"
            f"EarlyStop Counter: {early_stop_counter}/{patience}\n"  # 新增：显示早停计数器
            "--------------------------------------\n"
        )
        print(msg)
        f.write(msg)
        f.flush()

        # TensorBoard 记录
        tb_writer.add_scalar('train/loss', train_losses, epoch + 1)
        tb_writer.add_scalar('train/top1_acc', train_acces, epoch + 1)
        tb_writer.add_scalar('test/top1_acc', test_top1_acc, epoch + 1)
        tb_writer.add_scalar('test/top5_acc', test_top5_acc, epoch + 1)

    # 训练结束，输出最佳结果
    msg_best = f"Best Top-1 Acc: {best_acc:.2f}%\n"
    msg_top5_best = f"Best Top-5 Acc: {best_top5_acc:.2f}%\n"
    total_time = time.time() - since
    print('---------------------------')
    print(msg_best)
    print(msg_top5_best)
    print(f'Total Time: {total_time:.2f}s')
    f.write(msg_best)
    f.write(f'Total Time: {total_time:.2f}s\n')
    f.close()

if __name__ == "__main__":
    tb_path = "{}/{}/{}/{}".format(args.runs_path,args.dataset, args.model_names,  # 创建 TensorBoard 日志目录路径
                                     args.exp_postfix)
    tb_writer = SummaryWriter(log_dir=tb_path)
    lr = args.lr


    # 加载模型
    import timm
    if args.model_class=='resnet':
        # 加载 ResNet模型
        model = timm.create_model(args.model_names, pretrained=args.pre_trained, num_classes=args.classes_num)
    else:
        # 加载 ViT 模型
        model = timm.create_model(args.model_names, pretrained=args.pre_trained, num_classes=args.classes_num)


    if torch.cuda.is_available():
        model = model.cuda()
        print('GPU名称:',torch.cuda.get_device_name(0),'\n')

    optimizer = optim.SGD(  # 创建随机梯度下降 (SGD) 优化器
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)  # 创建余弦退火学习率调度器  自动调整lr

    train(model, optimizer, train_loader, test_loader, scheduler)  # 开始训练过程
