from typing import Tuple, Optional
import numpy as np
from tick.hawkes import HawkesExpKern

from .data_loader import load_raw_data
from .data_process import process_dataset


def create_decay_matrix(
    dim_process: int,
    base_value: float = 1.0,
    noise_scale: float = 0.01
) -> np.ndarray:
    """创建衰减矩阵（单位矩阵 + 高斯噪声）
    
    Args:
        dim_process: event type 数目
        base_value: 对角线基准值，默认1.0
        noise_scale: 噪声缩放系数，默认0.01
        
    Returns: np.ndarray: 衰减矩阵 (dim_process, dim_process)
    """
    if dim_process <= 0:
        raise ValueError(f"无效的维度 dim_process={dim_process}，必须为正整数")
    if base_value <= 0:
        raise ValueError(f"基准值必须为正数，当前 base_value={base_value}")

    identity = np.eye(dim_process) * base_value
    noise = noise_scale * np.random.randn(dim_process, dim_process)
    return identity + noise

def initialize_model(
    decay_matrix: np.ndarray,
    max_iter: int = 200,
    random_seed: Optional[int] = 42
) -> HawkesExpKern:
    """初始化霍克斯过程模型
    
    Args:
        decay_matrix: 衰减矩阵
        max_iter: 最大迭代次数，默认200
        random_seed: 随机种子，默认42（设为None禁用）
        
    Returns:
        HawkesExpKern: 初始化的霍克斯模型
        
    Raises:
        TypeError: 当输入矩阵类型错误时
    """
    if not isinstance(decay_matrix, np.ndarray):
        raise TypeError(f"衰减矩阵应为numpy数组，实际类型为 {type(decay_matrix)}")
    if decay_matrix.ndim != 2 or decay_matrix.shape[0] != decay_matrix.shape[1]:
        raise ValueError("衰减矩阵必须是方阵")
        
    if random_seed is not None:
        np.random.seed(random_seed)
    
    return HawkesExpKern(decays=decay_matrix, max_iter=max_iter)

def train_model(
    model: HawkesExpKern,
    train_data: list
) -> HawkesExpKern:
    """训练霍克斯过程模型
    
    Args:
        model: 初始化的模型对象
        train_data: 训练数据，格式参考tick文档
        verbose: 是否输出训练日志，默认True
        
    Returns:
        HawkesExpKern: 训练完成的模型
    """
           
    model.fit(train_data)
        
    return model

def evaluate_model(
    model: HawkesExpKern,
    datasets: dict
) -> Tuple[float, float, float]:
    """评估模型在不同数据集上的表现
    
    Args:
        model: 训练完成的模型
        datasets: 包含 train/valid/test 数据的字典
        verbose: 是否打印评估结果，默认True
        
    Returns:
        Tuple: 三个数据集的评分 (train_score, valid_score, test_score)
    """
    scores = []
    for split in ['train', 'valid', 'test']:
        data = datasets.get(split)
        if not data:
            raise ValueError(f"缺失 {split} 数据集")
            
        score = model.score(
            events=data,
            baseline=model.baseline,
            adjacency=model.adjacency
        )
        scores.append(score)
        
    return tuple(scores)

if __name__ == "__main__":

    # 参数配置
    MAX_ITER = 200

    raw_data = load_raw_data('earthquake')
    DIM_PROCESS = raw_data["train"]["dim_process"]
    
    # 1. 创建衰减矩阵
    decay_mat = create_decay_matrix(DIM_PROCESS)
    
    # 2. 初始化模型
    model = initialize_model(decay_mat, max_iter=MAX_ITER)
    
    # 3. 训练模型
    # train_data 应为 list of list of events 格式
    processed_data = process_dataset("earthquake")
    train_data = processed_data['train']['buckets']
    valid_data = processed_data['valid']['buckets']
    test_data = processed_data['test']['buckets']
    trained_model = train_model(model, train_data)
    
    # 4. 评估模型
    datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    scores = evaluate_model(trained_model, datasets)
    print(scores)
    