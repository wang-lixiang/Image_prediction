# 加载训练集和测试集
from Image_preprocessing import train_transform, test_transform

import os
from torchvision import datasets

# 数据集文件夹路径
dataset_dir = 'fruit30_split'
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')

# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)
# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)

# 各类别名称
class_names = train_dataset.classes
# 计算类别的总数
n_class = len(class_names)
# 映射关系：类别 到 索引号 train_dataset.class_to_idx
# 映射关系：索引号 到 类别
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
