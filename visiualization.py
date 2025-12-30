import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def load_hidden_states(file_path):
    """加载npz文件中的hidden states和labels"""
    data = np.load(file_path, allow_pickle=True)
    hidden_states = np.array(data["hidden_states"], dtype=np.float32)
    labels = np.array(data["labels"], dtype=np.int64)
    return hidden_states, labels


def clean_data(hidden_states, labels):
    """
    清理数据：移除包含NaN或Inf的样本
    
    Returns:
        cleaned_hidden_states, cleaned_labels, removed_count
    """
    # 检查NaN和Inf
    nan_mask = np.isnan(hidden_states).any(axis=1)
    inf_mask = np.isinf(hidden_states).any(axis=1)
    invalid_mask = nan_mask | inf_mask
    
    removed_count = invalid_mask.sum()
    
    if removed_count > 0:
        print(f"  Warning: Found {removed_count} samples with NaN/Inf values, removing them...")
        print(f"    - NaN samples: {nan_mask.sum()}")
        print(f"    - Inf samples: {inf_mask.sum()}")
    
    valid_mask = ~invalid_mask
    cleaned_hidden_states = hidden_states[valid_mask]
    cleaned_labels = labels[valid_mask]
    
    return cleaned_hidden_states, cleaned_labels, removed_count


def visualize_pca(hidden_states, labels, title="PCA Visualization", save_path=None):
    """
    对hidden states进行PCA降维并可视化
    """
    # 清理数据
    hidden_states, labels, removed = clean_data(hidden_states, labels)
    
    if len(hidden_states) == 0:
        print("Error: No valid samples after cleaning!")
        return
    
    # 标准化 - 使用更稳健的方式
    # 先处理常数列（方差为0的列）
    std = np.std(hidden_states, axis=0)
    constant_cols = std == 0
    if constant_cols.any():
        print(f"  Warning: Found {constant_cols.sum()} constant columns, replacing with 0...")
        hidden_states[:, constant_cols] = 0
        std[constant_cols] = 1  # 避免除以0
    
    # 手动标准化，避免除以0的问题
    mean = np.mean(hidden_states, axis=0)
    hidden_states_scaled = (hidden_states - mean) / (std + 1e-8)
    
    # 再次检查NaN
    if np.isnan(hidden_states_scaled).any():
        print("  Warning: NaN values after scaling, replacing with 0...")
        hidden_states_scaled = np.nan_to_num(hidden_states_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # PCA降维到2维
    pca = PCA(n_components=2)
    hidden_states_2d = pca.fit_transform(hidden_states_scaled)
    
    # 打印方差解释率
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 标签映射
    label_names = {0: "Negative", 1: "Positive"}
    colors = {0: "#E74C3C", 1: "#2ECC71"}
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    for label_id in np.unique(labels):
        mask = labels == label_id
        plt.scatter(
            hidden_states_2d[mask, 0],
            hidden_states_2d[mask, 1],
            c=colors.get(label_id, "#3498DB"),
            label=label_names.get(label_id, f"Class {label_id}"),
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
    
    plt.close()


def visualize_pca_3d(hidden_states, labels, title="PCA 3D Visualization", save_path=None):
    """
    对hidden states进行PCA降维到3维并可视化
    """
    # 清理数据
    hidden_states, labels, removed = clean_data(hidden_states, labels)
    
    if len(hidden_states) == 0:
        print("Error: No valid samples after cleaning!")
        return
    
    # 手动标准化
    std = np.std(hidden_states, axis=0)
    std[std == 0] = 1
    mean = np.mean(hidden_states, axis=0)
    hidden_states_scaled = (hidden_states - mean) / (std + 1e-8)
    hidden_states_scaled = np.nan_to_num(hidden_states_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # PCA降维到3维
    pca = PCA(n_components=3)
    hidden_states_3d = pca.fit_transform(hidden_states_scaled)
    
    print(f"  PCA 3D explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 标签映射
    label_names = {0: "Negative", 1: "Positive"}
    colors = {0: "#E74C3C", 1: "#2ECC71"}
    
    # 创建3D图表
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for label_id in np.unique(labels):
        mask = labels == label_id
        ax.scatter(
            hidden_states_3d[mask, 0],
            hidden_states_3d[mask, 1],
            hidden_states_3d[mask, 2],
            c=colors.get(label_id, "#3498DB"),
            label=label_names.get(label_id, f"Class {label_id}"),
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=10)
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)", fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
    
    plt.close()


def check_data_quality(hidden_states, labels):
    """检查数据质量"""
    print(f"  Data quality check:")
    print(f"    - Total samples: {len(hidden_states)}")
    print(f"    - Feature dim: {hidden_states.shape[1]}")
    print(f"    - NaN count: {np.isnan(hidden_states).sum()}")
    print(f"    - Inf count: {np.isinf(hidden_states).sum()}")
    print(f"    - Min value: {np.nanmin(hidden_states):.4f}")
    print(f"    - Max value: {np.nanmax(hidden_states):.4f}")
    print(f"    - Mean value: {np.nanmean(hidden_states):.4f}")


def main():
    # 设置路径
    hidden_states_dir = "data/hidden_states"
    output_dir = "data/pca_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有npz文件
    npz_files = [f for f in os.listdir(hidden_states_dir) if f.endswith('.npz')]
    
    for npz_file in npz_files:
        file_path = os.path.join(hidden_states_dir, npz_file)
        base_name = npz_file.replace(".npz", "")
        
        print(f"\nProcessing: {npz_file}")
        print("=" * 50)
        
        # 加载数据
        hidden_states, labels = load_hidden_states(file_path)
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 检查数据质量
        check_data_quality(hidden_states, labels)
        
        # 2D PCA可视化
        save_path_2d = os.path.join(output_dir, f"{base_name}_pca_2d.pdf")
        visualize_pca(
            hidden_states, 
            labels, 
            title=f"PCA 2D - {base_name}",
            save_path=save_path_2d
        )
        
        # 3D PCA可视化
        save_path_3d = os.path.join(output_dir, f"{base_name}_pca_3d.pdf")
        visualize_pca_3d(
            hidden_states, 
            labels, 
            title=f"PCA 3D - {base_name}",
            save_path=save_path_3d
        )
    
    print("\n" + "=" * 50)
    print("All visualizations completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()