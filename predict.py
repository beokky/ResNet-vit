import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL._imaging import display
from models import model_dict
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 加载训练好的模型
parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="resnet18", help="模型名称")
parser.add_argument("--classes_num", type=int, default=30, help="类别数")
parser.add_argument("--pre_trained", type=bool, default=False, help="是否使用预训练模型")
parser.add_argument("--model_path", type=str, default="model_pth/resnet18/evision2/best_34.pth", help="模型权重路径")
parser.add_argument("--img_path", type=str, default="dataset/test/1.jpg", help="图像路径")
args = parser.parse_args()
# 载入预训练图像分类模型
model = model_dict[args.model_names](num_classes=args.classes_num, pretrained=args.pre_trained)
# 加载模型权重
# 模型保存时保存了'epoch' 'model' 'acc'，因此加载时要提取出model
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model'])

model = model.eval()
model = model.to(device)


# 图像预处理
# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
# 载入一张测试图像
img_pil = Image.open(args.img_path)
input_img = test_transform(img_pil)  # 预处理
# 增加batch维度变为[1,3,224,224]
input_img = input_img.unsqueeze(0).to(device)

# 执行前向预测，得到所有类别的 logit 预测分数
pred_logits = model(input_img)
# 对 logit分数归一化转为0-1之间的概率，和为一
pred_softmax = F.softmax(pred_logits, dim=1)

# 预测结果分析
# 各类别置信度柱状图
plt.figure(figsize=(8,4))

x = range(30)
y = pred_softmax.cpu().detach().numpy()[0]

ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围

plt.xlabel('Class', fontsize=20)
plt.ylabel('Confidence', fontsize=20)
plt.tick_params(labelsize=16) # 坐标文字大小
plt.title(args.img_path, fontsize=25)
plt.show()

# 取置信度最大的 n 个结果
n = 10
top_n = torch.topk(pred_softmax, n)
# 解析出类别
pred_ids = top_n[1].cpu().detach().numpy().squeeze()

# 解析出置信度
confs = top_n[0].cpu().detach().numpy().squeeze()

# 载入数据集分类标签
df = pd.read_csv('fruit_class_index.csv',encoding='gbk')
idx_to_labels = {}
# 可以逐行遍历 DataFrame
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['class']]

# 图像分类结果写在原图上
# 读取图片
img_bgr = cv2.imread('dataset/test/1.jpg')

# 将 BGR 图像转换为 RGB 图像
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 使用 Pillow 来处理图像和绘制中文文本
pil_img = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(pil_img)

# 根据图片尺寸动态调整字体大小
img_width, img_height = pil_img.size
font_size = int(min(img_width, img_height) * 0.05)
# 选择一个支持中文的字体文件，比如微软雅黑
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)

for i in range(3):
    class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
    confidence = confs[i] * 100  # 获取置信度
    if i == 0:
        print('预测该水果是{}的可能性最大,可信度是{:>.2f}'.format(class_name, confidence))
    text = '{:<15} {:>.4f}'.format(class_name, confidence)


    # 绘制文本，指定位置，字体和颜色
    draw.text((25, 50 + 40 * i), text, font=font, fill=(255, 0, 0))

# 将图像转换回 OpenCV 格式
img_bgr = np.array(pil_img)
img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

# 保存图像
cv2.imwrite('output/img_pred.jpg', img_bgr)

# 载入预测结果图像
img_pred = Image.open('output/img_pred.jpg')


# # 预测结果表格输出
# pred_df = pd.DataFrame() # 预测结果表格
# for i in range(n):
#     class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
#     label_idx = int(pred_ids[i]) # 获取类别号
#     wordnet = idx_to_labels[pred_ids[i]][0] # 获取 WordNet
#     confidence = confs[i] * 100 # 获取置信度
#     new_row = pd.DataFrame(
#         [{'Class': class_name, 'Class_ID': label_idx, 'Confidence(%)': confidence, 'WordNet': wordnet}])
#
#     # 使用 pd.concat() 将 new_row 添加到 pred_df 中
#     pred_df = pd.concat([pred_df, new_row], ignore_index=True)
#     print(pred_df)