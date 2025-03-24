import os
import shutil
import random
from tqdm import tqdm

# 配置路径
data_dir = "dataset/Images"
output_dir = "dataset"
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# 获取类别信息
classes = sorted(os.listdir(data_dir))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# 写入标签文件
with open(os.path.join(output_dir, "labels.txt"), "w") as f:
    for cls, idx in class_to_idx.items():
        f.write(f"{cls} {idx}\n")


# 数据划分并复制文件
def split_and_copy_files():
    for cls in tqdm(classes, desc="Processing Classes"):
        class_path = os.path.join(data_dir, cls)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.endswith(".jpg")]
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        for subset, subset_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            subset_dir = os.path.join(output_dir, subset, cls)
            os.makedirs(subset_dir, exist_ok=True)
            for img in subset_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(subset_dir, img)
                shutil.copy(src, dst)


# 运行数据集划分
split_and_copy_files()
print("Dataset processing complete!")
