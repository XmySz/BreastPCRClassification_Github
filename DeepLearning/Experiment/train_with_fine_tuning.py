import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse

from models import ResNet18Encoder
from datasets import BreastPCRDataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

data_dir = "/media/ke/SSD_2T_2/HX/FormalTrainDataset/Images"     # 数据集路径
pretrained_path = "./checkpoints/FineTuning/model_best_auc.pth"  # 预训练权重路径


def evaluate(model, val_loader, criterion):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(val_loader), 100. * correct / total, auc


def train(fine_tune_mode=1):
    # 初始化tensorboard
    writer = SummaryWriter(f'runs/breast_pcr_experiment_mode{fine_tune_mode}')

    # 初始化模型
    model = ResNet18Encoder(in_channels=4).to(device)

    # 根据微调模式设置参数
    if fine_tune_mode == 1:
        # 加载预训练权重
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            print(f"成功加载预训练权重: {pretrained_path}")
        else:
            print(f"未找到预训练权重文件: {pretrained_path}")
        # 模式1: 微调所有参数
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("微调模式1: 微调所有模型参数")
    elif fine_tune_mode == 2:
        # 加载预训练权重
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            print(f"成功加载预训练权重: {pretrained_path}")
        else:
            print(f"未找到预训练权重文件: {pretrained_path}")
        # 模式2: 只微调分类头
        for param in model.parameters():
            param.requires_grad = False
        # 只允许分类头参数可训练
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        print("微调模式2: 只微调分类头参数")
    else:
        # 模式3: 从头训练
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.00005)
        print("模式3: 从头训练模型")

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 加载训练集和验证集
    train_dataset = BreastPCRDataset(data_dir, train=True)
    val_dataset = BreastPCRDataset(data_dir, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # 训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        train_labels = []
        train_probs = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for batch in progress_bar:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(probs[:, 1].detach().cpu().numpy())

            progress_bar.set_postfix({
                'train_loss': f'{total_loss / len(train_loader):.4f}',
                'train_acc': f'{100. * correct / total:.2f}%'
            })

        # 计算训练集指标
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_auc = roc_auc_score(train_labels, train_probs)

        # 验证阶段
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)

        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)

        # 打印每个epoch的结果
        print(f'\nEpoch {epoch + 1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.2f}%')
        print(f'Val AUC: {val_auc:.4f}')

        # 保存模型
        save_dir = f'checkpoints/mode{fine_tune_mode}'
        os.makedirs(save_dir, exist_ok=True)

        # 保存验证集性能最好的模型
        if epoch == 0:
            best_val_auc = val_auc
            best_val_acc = val_acc
        else:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_path = f'{save_dir}/model_best_auc.pth'
                torch.save(model.state_dict(), best_path)
                print(f'New best AUC model saved to {best_path}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = f'{save_dir}/model_best_acc.pth'
                torch.save(model.state_dict(), best_path)
                print(f'New best accuracy model saved to {best_path}')

    writer.close()


if __name__ == "__main__":
    train(fine_tune_mode=2)
