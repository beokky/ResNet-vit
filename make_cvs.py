import os
import csv

# 数据集路径
data_dir = 'dataset/train'
output_csv = 'fruit_class_index.csv'

# 获取类别名称并排序，保证索引一致
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# 打开 CSV 文件准备写入
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['index', 'class_id', 'class_name'])

    # 为每个类别生成索引和 class_id
    for idx, class_name in enumerate(class_names):
        class_id = f"fr_{class_name.lower()}"  # 给每个类别一个唯一的 class_id，比如 'fr_apple'
        writer.writerow([idx, class_id, class_name])

print(f"分类索引 CSV 文件已保存到 {output_csv}")
