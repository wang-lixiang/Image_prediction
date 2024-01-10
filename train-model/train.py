import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

from Data_Loader import train_loader, test_loader
from Load_model import model, optimizer

# 获取计算硬件
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练轮次 Epoch
EPOCHS = 30

# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_one_batch(images, labels):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''

    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    # log_train['train_precision'] = precision_score(labels, preds, average='macro')
    # log_train['train_recall'] = recall_score(labels, preds, average='macro')
    # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train


def evaluate_testset():
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''

    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

    return log_test


if __name__ == '__main__':
    epoch = 0
    batch_idx = 0
    best_test_accuracy = 0

    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    log_train = {}
    log_train['epoch'] = 0
    log_train['batch'] = 0
    images, labels = next(iter(train_loader))
    log_train.update(train_one_batch(images, labels))
    df_train_log = df_train_log._append(log_train, ignore_index=True)

    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['epoch'] = 0
    log_test.update(evaluate_testset())
    df_test_log = df_test_log._append(log_test, ignore_index=True)

    for epoch in range(1, EPOCHS + 1):

        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        model.train()
        for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
            batch_idx += 1
            log_train = train_one_batch(images, labels)
            df_train_log = df_train_log._append(log_train, ignore_index=True)

        lr_scheduler.step()

        ## 测试阶段
        model.eval()
        log_test = evaluate_testset()
        df_test_log = df_test_log._append(log_test, ignore_index=True)

        # 保存最新的最佳模型文件
        if log_test['test_accuracy'] > best_test_accuracy:
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
            torch.save(model, new_best_checkpoint_path)
            print('保存新的最佳模型', 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy))
            # best_test_accuracy = log_test['test_accuracy']

    df_train_log.to_csv('训练日志-训练集.csv', index=False)
    df_test_log.to_csv('训练日志-测试集.csv', index=False)
