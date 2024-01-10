# 载入模型，迁移训练
from Load_Dataset import n_class

from torchvision import models
import torch.optim as optim
import torch.nn as nn

model = models.resnet18(pretrained=True)  # 载入预训练模型
# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, n_class)
# 只微调训练最后一层全连接层的参数，其它层冻结
optimizer = optim.Adam(model.fc.parameters())
