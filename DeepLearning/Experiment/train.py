import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np

from models import ResNet18Encoder
from datasets import BreastPCRDataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCHS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# 数据集路径
data_dir = r"/media/ke/SSD_2T_2/HX/Dataset/ISPY2_SELECT/Cropped_OnlyROI_3_TrainAndValid"
test_dir = r'/media/ke/SSD_2T_2/HX/Dataset/center2/Cropped_OnlyROI_3_ExternalTest'


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


def train():
    # 初始化tensorboard
    writer = SummaryWriter('runs/breast_pcr_experiment')

    # 初始化模型
    model = ResNet18Encoder(in_channels=1).to(device)

    # 加载训练集和验证集
    train_dataset = BreastPCRDataset(data_dir, train=True)
    val_dataset = BreastPCRDataset(data_dir, train=False)
    test_dataset = BreastPCRDataset(test_dir, train=False)

    # 计算每个类别的样本数量并创建权重
    class_counts = torch.bincount(torch.tensor([int(sample["label"]) for sample in train_dataset]))
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum() * len(class_counts)
    weights = weights.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.15
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.00005)

    # 添加学习率调度器,使学习率线性减小
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=EPOCHS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE // 2,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE // 2,
        shuffle=False,
        num_workers=4,
        drop_last=True
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

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # 计算训练集指标
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_auc = roc_auc_score(train_labels, train_probs)

        # 验证阶段
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)

        # 外部测试阶段
        test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)

        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('AUC/test', test_auc, epoch)

        # 打印每个epoch的结果
        print(f'\nEpoch {epoch + 1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Val Accuracy: {val_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Train AUC: {train_auc:.4f}')
        print(f'Val AUC: {val_auc:.4f}')
        print(f'Test AUC: {test_auc:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # 保存模型
        os.makedirs('Checkpoints', exist_ok=True)

        # 保存验证集性能最好的模型
        if epoch == 0:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_test_auc = test_auc
            best_test_acc = test_acc
        else:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_path = 'Checkpoints/model_best_val_auc.pth'
                torch.save(model.state_dict(), best_path)
                print(f'New best validation AUC model saved to {best_path}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = 'Checkpoints/model_best_val_acc.pth'
                torch.save(model.state_dict(), best_path)
                print(f'New best validation accuracy model saved to {best_path}')

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_path = 'Checkpoints/model_best_test_auc.pth'
                torch.save(model.state_dict(), best_path)
                print(f'New best test AUC model saved to {best_path}')

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_path = 'Checkpoints/model_best_test_acc.pth'
                torch.save(model.state_dict(), best_path)
                print(f'New best test accuracy model saved to {best_path}')

    writer.close()


if __name__ == "__main__":
    train()
