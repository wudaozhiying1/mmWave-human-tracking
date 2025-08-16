import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

class EdgeConv(nn.Module):
    """
    EdgeConv模块 - 基于论文实现
    用于提取点云的空间几何特征
    """
    def __init__(self, in_channels, out_channels, k=10):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
    def knn_graph(self, x, k):
        """构建KNN图"""
        batch_size, num_dims, num_points = x.size()
        
        # 计算点对之间的距离
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (batch_size, num_points, num_points)
        xx = torch.sum(x**2, dim=1, keepdim=True)  # (batch_size, 1, num_points)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        # 获取k个最近邻
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx
    
    def get_graph_feature(self, x, k, idx=None):
        """构建局部有向邻域图特征"""
        batch_size, num_dims, num_points = x.size()
        device = x.device  # 使用输入张量的设备
        
        if idx is None:
            idx = self.knn_graph(x, k)  # (batch_size, num_points, k)
        
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        # 连接中心点和邻居点的特征
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature  # (batch_size, 2*num_dims, num_points, k)
    
    def forward(self, x):
        """前向传播"""
        # x: (batch_size, in_channels, num_points)
        x = self.get_graph_feature(x, self.k)  # (batch_size, 2*in_channels, num_points, k)
        x = self.leaky_relu(self.bn1(self.conv1(x)))  # (batch_size, 64, num_points, k)
        x = self.leaky_relu(self.bn2(self.conv2(x)))  # (batch_size, out_channels, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, out_channels, num_points)
        return x

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=25):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    """Transformer编码器模块"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src):
        # src: (batch_size, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)
        return output.transpose(0, 1)  # (batch_size, seq_len, d_model)

class PETerNetwork(nn.Module):
    """
    PETer网络 - Point EdgeConv and Transformer
    基于论文实现的完整网络架构
    """
    def __init__(self, num_classes=3, num_points=100, num_frames=25, k=10):
        super(PETerNetwork, self).__init__()
        self.num_points = num_points
        self.num_frames = num_frames
        self.k = k
        
        # EdgeConv模块
        self.edge_conv1 = EdgeConv(4, 64, k)  # 输入：xyz + intensity
        self.edge_conv2 = EdgeConv(64, 128, k)
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 全局最大池化
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=256,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.3
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        x: (batch_size, num_frames, num_points, 4)  # xyz + intensity
        """
        batch_size, num_frames, num_points, _ = x.size()
        
        # 逐帧处理
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i, :, :].transpose(1, 2)  # (batch_size, 4, num_points)
            
            # EdgeConv特征提取
            feat = self.edge_conv1(frame)  # (batch_size, 64, num_points)
            feat = self.edge_conv2(feat)   # (batch_size, 128, num_points)
            
            # MLP增强特征
            feat = self.mlp(feat)  # (batch_size, 256, num_points)
            
            # 全局最大池化
            global_feat = self.global_pool(feat).squeeze(-1)  # (batch_size, 256)
            frame_features.append(global_feat)
        
        # 堆叠帧特征
        sequence_features = torch.stack(frame_features, dim=1)  # (batch_size, num_frames, 256)
        
        # Transformer时序建模
        temporal_features = self.transformer(sequence_features)  # (batch_size, num_frames, 256)
        
        # 全局时序池化
        global_temporal = temporal_features.max(dim=1)[0]  # (batch_size, 256)
        
        # 分类
        output = self.classifier(global_temporal)  # (batch_size, num_classes)
        
        return output

class DataProcessor:
    """数据预处理器 - 基于论文方法"""
    
    def __init__(self, num_points=100, num_frames=25, fusion_frames=6):
        self.num_points = num_points
        self.num_frames = num_frames
        self.fusion_frames = fusion_frames
    
    def filter_by_range(self, points, radar_type='TI'):
        """动态干扰滤除 - 基于实际数据范围调整"""
        # 基于观察到的数据范围，使用更宽松的过滤
        # 实际数据: X: [-14.269, 10.232], Y: [-0.248, 13.545], Z: [-11.730, 12.765]
        
        # 过滤明显的异常值，但保留大部分有效数据
        mask = (
            (points[:, 0] >= -10.0) & (points[:, 0] <= 10.0) &  # x
            (points[:, 1] >= -2.0) & (points[:, 1] <= 15.0) &   # y
            (points[:, 2] >= -8.0) & (points[:, 2] <= 10.0)     # z
        )
        
        return points[mask]
    
    def dbscan_clustering(self, points, eps=2.0, min_samples=2):
        """DBSCAN聚类去除噪声点 - 使用更宽松的参数"""
        from sklearn.cluster import DBSCAN
        
        if len(points) < min_samples:
            return points
        
        # 只对xyz坐标进行聚类，使用更大的eps值
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
        
        # 保留最大簇的点
        labels = clustering.labels_
        if len(set(labels)) > 1:  # 存在有效簇
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster = unique_labels[np.argmax(counts)]
                mask = labels == largest_cluster
                return points[mask]
        
        return points
    
    def multi_frame_fusion(self, frame_list):
        """多帧融合"""
        if len(frame_list) == 0:
            return np.array([]).reshape(0, 4)
        
        # 简单拼接所有帧的点
        fused_points = np.vstack([frame for frame in frame_list if len(frame) > 0])
        return fused_points
    
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
            # 返回零填充的点云
            return np.zeros((num_points, 4))
        
        if len(points) >= num_points:
            # 随机采样
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices]
        else:
            # 重复采样
            indices = np.random.choice(len(points), num_points, replace=True)
            return points[indices]
    
    def create_sequences(self, df, action_label, radar_type='TI'):
        """创建时序序列数据"""
        sequences = []
        labels = []
        
        # 按文件分组
        files = df['file_name'].unique()
        
        for file_name in files:
            file_data = df[df['file_name'] == file_name].sort_values('frame_num')
            frames = file_data.groupby('frame_num')
            
            # 准备帧数据
            frame_list = []
            for frame_num, frame_points in frames:
                if len(frame_points) > 0:
                    points = frame_points[['x', 'y', 'z']].values
                    # 添加强度信息（如果没有则设为1）
                    intensity = np.ones((len(points), 1))
                    points_with_intensity = np.hstack([points, intensity])
                    
                    # 数据预处理
                    points_with_intensity = self.filter_by_range(points_with_intensity, radar_type)
                    
                    if len(points_with_intensity) > 0:
                        frame_list.append(points_with_intensity)
            
            # 多帧融合处理
            fused_frames = []
            for i in range(0, len(frame_list), self.fusion_frames):
                fusion_group = frame_list[i:i+self.fusion_frames]
                if len(fusion_group) > 0:
                    fused_points = self.multi_frame_fusion(fusion_group)
                    
                    # 聚类去噪
                    if len(fused_points) > 0:
                        fused_points = self.dbscan_clustering(fused_points)
                        
                        # 标准化和采样
                        fused_points = self.normalize_point_cloud(fused_points)
                        fused_points = self.sample_points(fused_points, self.num_points)
                        fused_frames.append(fused_points)
            
            # 创建固定长度序列
            if len(fused_frames) >= self.num_frames:
                for i in range(len(fused_frames) - self.num_frames + 1):
                    sequence = np.array(fused_frames[i:i+self.num_frames])
                    sequences.append(sequence)
                    labels.append(action_label)
        
        return np.array(sequences), np.array(labels)

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_peter_network():
    """测试PETer网络"""
    print("=== Testing PETer Network ===")
    
    # 创建模型
    model = PETerNetwork(num_classes=3, num_points=100, num_frames=25, k=10)
    
    # 打印模型信息
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model size: {count_parameters(model) / 1e6:.2f}M")
    
    # 测试输入
    batch_size = 2
    num_frames = 25
    num_points = 100
    
    # 创建随机输入数据 (batch_size, num_frames, num_points, 4)
    x = torch.randn(batch_size, num_frames, num_points, 4)
    
    print(f"Input shape: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("PETer network test completed successfully!")

if __name__ == "__main__":
    test_peter_network() 