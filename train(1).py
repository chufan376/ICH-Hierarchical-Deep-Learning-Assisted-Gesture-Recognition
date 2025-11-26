import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms  # 若处理图像数据
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()


# -------------------------- 1. 数据加载与预处理 --------------------------
class CustomDataset(Dataset):
    """自定义数据集类，根据任务修改"""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: 输入特征数据（如图像数组、数值特征等）
            labels: 标签数据（整数类型，对应类别）
            transform: 数据预处理函数（如标准化、数据增强等）
        """
        self.data = torch.tensor(data.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # 若有预处理，应用变换（如图像数据的标准化）
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def get_data_loaders(data, labels, batch_size=32, val_split=0.2, transform=None):
    """
    生成训练集、验证集的数据加载器
    Args:
        data: 全部特征数据
        labels: 全部标签数据
        batch_size: 批次大小
        val_split: 验证集占比
        transform: 数据预处理函数
    Returns:
        train_loader, val_loader: 训练集和验证集的数据加载器
    """
    # 实例化数据集
    dataset = CustomDataset(data, labels, transform=transform)
    #print(dataset.__len__())
    #print(dataset.__getitem__(0))

    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 生成数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # 多线程加载数据
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


# -------------------------- 2. 模型定义 --------------------------
class Classifier(nn.Module):
    """分类神经网络模型，根据任务修改网络结构"""

    def __init__(self, input_dim, num_classes, hidden_dims=[256,128,64,32]):
        """
        Args:
            input_dim: 输入特征维度（如28*28=784 for MNIST）
            num_classes: 类别数量（二分类为1，多分类为类别数）
            hidden_dims: 隐藏层维度列表
        """
        super(Classifier, self).__init__()

        # 构建隐藏层
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),  # 激活函数
                nn.BatchNorm1d(hidden_dim),  # 批标准化
                nn.Dropout(0.3)  # Dropout防止过拟合
            ])
            in_dim = hidden_dim

        # 输出层（二分类用sigmoid，多分类用softmax）
        self.classifier = nn.Sequential(
            *layers,
            nn.Linear(in_dim, num_classes)
        )

        # 根据任务选择输出激活函数
        self.output_activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # 展平输入（若输入为图像等多维数据）
        x = x.view(x.size(0), -1)  # (batch_size, input_dim)
        logits = self.classifier(x)
        probs = self.output_activation(logits)
        return logits, probs


# -------------------------- 3. 训练与验证函数 --------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()  # 切换到训练模式
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        logits, probs = model(inputs)

        # 计算损失（二分类需注意标签格式）
        if model.output_activation.__class__.__name__ == 'Sigmoid':
            # 二分类：标签转为float，损失用BCEWithLogitsLoss更稳定
            loss = criterion(logits.squeeze(), labels.float())
            preds = (probs.squeeze() > 0.5).float()  # 阈值判断
        else:
            # 多分类：标签为long，损失用CrossEntropyLoss
            loss = criterion(logits, labels.squeeze(dim=1).long())
            preds = torch.argmax(probs, dim=1)  # 取概率最大的类别

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 累计损失和预测结果
        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            logits, probs = model(inputs)

            # 计算损失
            if model.output_activation.__class__.__name__ == 'Sigmoid':
                loss = criterion(logits.squeeze(), labels.float())
                preds = (probs.squeeze() > 0.5).float()
            else:
                loss = criterion(logits, labels.squeeze(dim=1).long())
                preds = torch.argmax(probs, dim=1)

            # 累计损失和预测结果
            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均损失、准确率和详细指标
    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, output_dict=True)
    return avg_loss, acc, report


# -------------------------- 4. 主训练流程 --------------------------
def main(data, labels, input_dim, num_classes, config):
    """
    主函数：训练并验证模型
    Args:
        data: 输入特征数据
        labels: 标签数据
        input_dim: 输入特征维度
        num_classes: 类别数量
        config: 训练配置字典（包含学习率、批次大小等）
    """
    # 设备配置（GPU优先）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据预处理（根据任务修改，如图像标准化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor（若输入为图像）
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化（示例）
    ]) if config.get('use_transform', True) else None

    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        data=data,
        labels=labels,
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        transform=transform
    )

    # 初始化模型、损失函数和优化器
    model = Classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=config['hidden_dims']
    ).to(device)

    # 根据任务选择损失函数
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()  # 二分类
    else:
        criterion = nn.CrossEntropyLoss()  # 多分类

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']  # L2正则化
    )

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5
    )

    # 记录训练过程
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc, val_report = validate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])
            print(f"Epoch {epoch + 1}: Best model saved (val_acc: {val_acc:.4f})")

        # 打印日志
        print(
            f"Epoch {epoch + 1}/{config['epochs']}\n"
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
            f"-------------------------"
        )

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

    print("Training complete!")
    return model, history, val_report


# -------------------------- 5. 使用示例 --------------------------
if __name__ == "__main__":
    # 示例：生成随机数据（实际使用时替换为你的数据）
    # 假设输入维度为20，1000个样本，3分类任务
    input_dim = 205
    num_classes = 24
    s = pd.read_csv(r'./resultce.csv', encoding='utf-8-sig')
    data = s.iloc[:,:205]  # 特征DataFrame
    labels = s.iloc[:,205:]  # 标签Series

    # 训练配置
    config = {
        'batch_size': 32,
        'val_split': 0.2,
        'hidden_dims': [256, 128,64,32],  # 隐藏层维度
        'lr': 0.01,  # 学习率
        'weight_decay': 1e-5,  # L2正则化系数
        'epochs': 100,  # 训练轮数
        'save_path': r'./best_model1.pth',  # 最佳模型保存路径
        'use_transform': False  # 若为图像数据可开启
    }

    # 启动训练
    model, history, val_report = main(
        data=data,
        labels=labels,
        input_dim=input_dim,
        num_classes=num_classes,
        config=config
    )

    # 打印验证集详细指标
    print("\nValidation Report:")
    print(val_report)
