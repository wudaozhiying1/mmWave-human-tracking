#!/usr/bin/env python3
"""
基于论文方法的PETer模型训练脚本
使用sit_data, squat_data, stand_data三组数据进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from peter_network import PETerNetwork, DataProcessor
import warnings
warnings.filterwarnings('ignore')

class RadarPointCloudDataset(Dataset):
    """雷达点云数据集类"""
    
    def __init__(self, data_paths, labels, sample_indices, num_points=100, num_frames=25, transform=None, augment=False):
        """
        初始化数据集
        Args:
            data_paths: 数据文件路径列表
            labels: 对应的标签列表
            sample_indices: 样本索引列表
            num_points: 每帧采样点数
            num_frames: 每个样本的帧数
            transform: 数据变换
            augment: 是否进行数据增强
        """
        self.data_paths = data_paths
        self.labels = labels
        self.sample_indices = sample_indices
        self.num_points = num_points
        self.num_frames = num_frames
        self.transform = transform
        self.augment = augment
        
        # 标签映射
        self.label_to_idx = {'sit': 0, 'squat': 1, 'stand': 2}
        self.idx_to_label = {0: 'sit', 1: 'squat', 2: 'stand'}
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # 加载数据
        with open(self.data_paths[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取指定索引的样本
        sample_idx = self.sample_indices[idx]
        sample = data[sample_idx]
        
        # 获取目标点云数据
        target_points = sample['target_points']
        if len(target_points) == 0:
            # 如果没有目标点，创建零填充数据
            target_points = np.zeros((self.num_points, 3))
        else:
            target_points = np.array(target_points)
        
        # 标准化点云
        target_points = self.normalize_point_cloud(target_points)
        
        # 采样固定数量的点
        target_points = self.sample_points(target_points, self.num_points)
        
        # 数据增强（仅对训练集）
        if self.augment:
            target_points = self.augment_point_cloud(target_points)
        
        # 添加强度信息（如果没有则设为1）
        intensity = np.ones((len(target_points), 1))
        target_points = np.hstack([target_points, intensity])
        
        # 创建时序序列（这里简化为单帧，实际应该有多帧）
        # 为了适配模型，我们重复单帧数据
        sequence = np.tile(target_points, (self.num_frames, 1, 1))
        
        # 转换为张量
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([self.label_to_idx[self.labels[idx]]])
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label
    
    def normalize_point_cloud(self, points):
        """标准化点云"""
        if len(points) == 0:
            return points
        
        # 中心化
        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] = points[:, :3] - centroid
        
        # 缩放到单位球
        max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
        if max_dist > 0:
            points[:, :3] = points[:, :3] / max_dist
        
        return points
    
    def sample_points(self, points, num_points):
        """固定采样点数"""
        if len(points) == 0:
            return np.zeros((num_points, 3))
        
        if len(points) >= num_points:
            # 随机采样
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices]
        else:
            # 重复采样
            indices = np.random.choice(len(points), num_points, replace=True)
            return points[indices]
    
    def augment_point_cloud(self, points):
        """点云数据增强"""
        if len(points) == 0:
            return points
        
        # 随机旋转
        if np.random.random() < 0.5:
            angle = np.random.uniform(-0.1, 0.1)  # 小角度旋转
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            points[:, :3] = points[:, :3] @ rotation_matrix.T
        
        # 随机噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, points[:, :3].shape)
            points[:, :3] += noise
        
        # 随机缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale
        
        return points

def load_training_data():
    """加载训练数据"""
    print("📊 加载训练数据...")
    
    data_paths = []
    labels = []
    sample_indices = []
    
    # 数据目录配置
    data_config = {
        'sit': 'radar_data/sit_improved_fused/improved_fused_pointcloud',
        'squat': 'radar_data/squat_improved_fused/improved_fused_pointcloud', 
        'stand': 'radar_data/stand_improved_fused/improved_fused_pointcloud'
    }
    
    for action, data_dir in data_config.items():
        if os.path.exists(data_dir):
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            print(f"   📁 {action}: {len(json_files)} 个文件")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        # 每个样本作为一个数据点
                        for i, sample in enumerate(data):
                            if isinstance(sample, dict) and 'target_points' in sample:
                                data_paths.append(json_file)
                                labels.append(action)
                                sample_indices.append(i)
                except Exception as e:
                    print(f"   ❌ 读取失败 {json_file}: {e}")
    
    print(f"📊 总数据点: {len(data_paths)}")
    
    # 统计标签分布
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"📊 标签分布: {label_counts}")
    
    return data_paths, labels, sample_indices

def create_data_loaders(data_paths, labels, sample_indices, batch_size=32, test_size=0.2, val_size=0.1):
    """创建数据加载器"""
    print("🔧 创建数据加载器...")
    
    # 划分训练集、验证集、测试集
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        data_paths, labels, sample_indices, test_size=test_size, stratify=labels, random_state=42
    )
    
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=val_size, stratify=y_temp, random_state=42
    )
    
    print(f"📊 训练集: {len(X_train)} 个样本")
    print(f"📊 验证集: {len(X_val)} 个样本") 
    print(f"📊 测试集: {len(X_test)} 个样本")
    
    # 创建数据集（训练集使用数据增强）
    train_dataset = RadarPointCloudDataset(X_train, y_train, idx_train, augment=True)
    val_dataset = RadarPointCloudDataset(X_val, y_val, idx_val, augment=False)
    test_dataset = RadarPointCloudDataset(X_test, y_test, idx_test, augment=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, y_train

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cpu', class_weights=None):
    """训练模型"""
    print("🚀 开始训练模型...")
    
    # 移动到设备
    model = model.to(device)
    
    # 损失函数和优化器
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"📊 使用加权损失函数，权重: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("📊 使用标准损失函数")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 学习率调度
        scheduler.step()
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_peter_model.pth')
            print(f'  💾 保存最佳模型 (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  ⏹️  早停训练 (Patience: {patience})')
                break
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    print("📊 评估模型...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算准确率
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    print(f'📊 测试准确率: {accuracy:.2f}%')
    
    # 分类报告
    print('\n📋 分类报告:')
    print(classification_report(all_targets, all_predictions, 
                               target_names=['sit', 'squat', 'stand']))
    
    return all_predictions, all_targets, accuracy

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_class_weights(labels):
    """计算类别权重以处理数据不平衡"""
    from collections import Counter
    
    # 统计每个类别的样本数
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # 计算权重（样本数越少，权重越大）
    weights = []
    for i in range(3):  # 3个类别：sit, squat, stand
        class_name = ['sit', 'squat', 'stand'][i]
        if class_name in class_counts:
            # 使用逆频率权重
            weight = total_samples / (len(class_counts) * class_counts[class_name])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    print(f"📊 类别权重: {weights}")
    return weights

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['sit', 'squat', 'stand'],
                yticklabels=['sit', 'squat', 'stand'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_training_summary(class_weights, accuracy):
    """打印训练总结"""
    print("\n" + "="*60)
    print("🎯 训练总结")
    print("="*60)
    print(f"📊 类别权重: {class_weights}")
    print(f"📈 最终测试准确率: {accuracy:.2f}%")
    print("🔧 使用的技术:")
    print("   ✅ 加权损失函数 (处理数据不平衡)")
    print("   ✅ 数据增强 (旋转、噪声、缩放)")
    print("   ✅ 早停机制 (防止过拟合)")
    print("   ✅ 学习率调度 (优化收敛)")
    print("   ✅ PETer网络架构 (EdgeConv + Transformer)")
    print("="*60)

def main():
    """主函数"""
    print("🎯 PETer模型训练开始")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    data_paths, labels, sample_indices = load_training_data()
    
    if len(data_paths) == 0:
        print("❌ 没有找到有效数据")
        return
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, y_train = create_data_loaders(data_paths, labels, sample_indices)
    
    # 计算类别权重以处理数据不平衡
    class_weights = calculate_class_weights(y_train)
    
    # 创建模型
    model = PETerNetwork(num_classes=3, num_points=100, num_frames=25, k=10)
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型（使用加权损失）
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device=device, class_weights=class_weights
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_peter_model.pth'))
    
    # 评估模型
    predictions, targets, accuracy = evaluate_model(model, test_loader, device)
    
    # 绘制结果
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(targets, predictions)
    
    # 打印训练总结
    print_training_summary(class_weights, accuracy)
    
    print("✅ 训练完成!")

if __name__ == "__main__":
    main() 