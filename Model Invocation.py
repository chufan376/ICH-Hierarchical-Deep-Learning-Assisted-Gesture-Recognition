import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from train import Classifier  # 导入你的模型类（如之前的Classifier）

# -------------------------- 1. 配置基础参数 --------------------------
# 模型相关参数（需与训练时保持一致！）
INPUT_DIM = 205  # 输入特征维度（如训练时的input_dim）
NUM_CLASSES = 24  # 类别数量（如训练时的num_classes）
HIDDEN_DIMS = [256,128,64,32]  # 隐藏层维度（需与训练时完全相同）
MODEL_PATH = "./best_model.pth"  # 训练好的模型权重文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备（GPU/CPU）

# -------------------------- 2. 定义预测数据加载类 --------------------------
class PredictDataset(Dataset):
    """预测专用数据集类，处理待预测的特征数据"""

    def __init__(self, data, transform=None):
        """
        Args:
            data: 待预测的特征数据（DataFrame/NumPy数组/Tensor）
            transform: 数据预处理函数（需与训练时保持一致）
        """
        # 数据类型转换：统一转为float32的Tensor（避免 dtype 不匹配）
        if isinstance(data, pd.DataFrame):
            self.data = torch.tensor(data.iloc[:,:205].to_numpy(), dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            self.data = data.to(dtype=torch.float32)
        else:
            raise TypeError("数据格式不支持，需为DataFrame/NumPy数组/Tensor")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample  # 预测时无需标签，仅返回特征


# -------------------------- 3. 加载模型并初始化 --------------------------
def load_trained_model(model_path, input_dim, num_classes, hidden_dims, device):
    """
    加载训练好的模型权重，初始化模型

    Args:
        model_path: 模型权重文件路径
        input_dim: 输入特征维度
        num_classes: 类别数量
        hidden_dims: 隐藏层维度
        device: 运行设备

    Returns:
        model: 加载好权重的模型（评估模式）
    """
    # 1. 初始化模型结构（需与训练时完全一致）
    model = Classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims
    ).to(device)

    # 2. 加载训练好的权重（避免因设备不匹配报错）
    # 若训练时用GPU，预测时用CPU，需添加 map_location=device
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    # 3. 切换为评估模式（关闭Dropout、BatchNorm的训练行为）
    model.eval()
    return model


# -------------------------- 4. 核心预测函数 --------------------------
def predict(model, predict_loader, device, num_classes):
    """
    对数据进行预测，返回预测类别和类别概率

    Args:
        model: 加载好的模型（评估模式）
        predict_loader: 待预测数据的DataLoader
        device: 运行设备
        num_classes: 类别数量

    Returns:
        all_pred_classes: 所有样本的预测类别（整数索引）
        all_pred_probs: 所有样本的类别概率分布（每行和为1）
    """
    all_pred_classes = []
    all_pred_probs = []

    # 关闭梯度计算（预测时无需反向传播，提升速度）
    with torch.no_grad():
        for inputs in predict_loader:
            # 数据送设备
            inputs = inputs.to(device)

            # 前向传播：获取模型输出（logits为未激活值，probs为概率）
            logits, probs = model(inputs)

            # 转换为numpy数组（便于后续处理）
            probs_np = probs.cpu().numpy()  # 概率分布：[batch_size, num_classes]

            # 确定预测类别（多分类取概率最大的索引，二分类用0.5阈值）
            if num_classes == 1:  # 二分类
                pred_classes = (probs_np > 0.5).astype(int).squeeze()
            else:  # 多分类
                pred_classes = np.argmax(probs_np, axis=1)  # 取概率最大的类别索引

            # 收集结果
            all_pred_classes.extend(pred_classes)
            all_pred_probs.extend(probs_np)

    # 转为numpy数组（方便后续分析，如保存到CSV）
    return np.array(all_pred_classes), np.array(all_pred_probs)


def open_image_by_filename(filename):
    """
    根据文件名打开对应的图片

    参数:
        filename: 图片文件名（可包含路径，如"images/hand_01.jpg"）
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"错误：文件 '{filename}' 不存在")
        return

    # 检查文件是否为图片格式（简单判断扩展名）
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    if not filename.lower().endswith(valid_extensions):
        print(f"错误：文件 '{filename}' 不是支持的图片格式（支持{valid_extensions}）")
        return

    try:
        # 打开图片并显示
        with Image.open(filename) as img:
            img.show()  # 调用系统默认图片查看器显示
            print(f"成功打开图片：{filename}（尺寸：{img.size[0]}x{img.size[1]}）")
    except Exception as e:
        print(f"打开图片失败：{str(e)}")
# -------------------------- 5. 完整预测流程示例 --------------------------
if __name__ == "__main__":
    t = ['0', '1', '2', '7', '6', '5', '3', '4', '8', '9', '60',
     '50', '70', '40', '80', '20', '90', '30', 'all right', 'good',
     'hold the beaker', 'i love you', 'thank you', 'ok']
    # -------------------------- 步骤1：准备待预测数据 --------------------------
    # 示例1：从CSV读取待预测数据（假设CSV中仅包含特征列，无标签列）
    predict_data_df = pd.read_csv("./resultce2.csv",encoding='utf-8-sig')  # 待预测数据（DataFrame）

    # 示例2：若用单样本测试（手动构造特征）
    # single_sample = np.array([[1.2, 3.4, 2.1, ..., 0.8]])  # 1个样本，INPUT_DIM个特征
    # predict_data_df = pd.DataFrame(single_sample)

    # 数据预处理（需与训练时完全一致！如标准化）
    # 示例：若训练时用了Normalize，预测时必须用相同的均值和标准差
    from torchvision import transforms

    predict_transform = transforms.Compose([
        # 若训练时无额外预处理，可留空；若有，需完全对应（如标准化）
        # transforms.Normalize(mean=[0.5], std=[0.5])  # 示例：与训练时一致的标准化
    ])

    # -------------------------- 步骤2：创建预测数据加载器 --------------------------
    predict_dataset = PredictDataset(
        data=predict_data_df,
        transform=predict_transform
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=32,  # 批次大小可根据设备内存调整，不影响结果
        shuffle=False,  # 预测时无需打乱
        num_workers=2
    )

    # -------------------------- 步骤3：加载模型 --------------------------
    trained_model = load_trained_model(
        model_path=MODEL_PATH,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dims=HIDDEN_DIMS,
        device=DEVICE
    )
    print(f"模型已加载至 {DEVICE}，准备预测...")

    # -------------------------- 步骤4：执行预测 --------------------------
    pred_classes, pred_probs = predict(
        model=trained_model,
        predict_loader=predict_loader,
        device=DEVICE,
        num_classes=NUM_CLASSES
    )

    # -------------------------- 步骤5：处理预测结果 --------------------------
    # 1. （可选）添加类别名称映射（若类别索引对应实际名称，如0→'cat'，1→'dog'）
    #class_name_mapping = {0: "ClassA", 1: "ClassB", 2: "ClassC"}  # 需根据实际任务修改
    #pred_class_names = [class_name_mapping[cls] for cls in pred_classes]

    # 2. 整理结果为DataFrame（便于保存或查看）
    result_df = predict_data_df.copy()  # 保留原始特征
    result_df["预测类别索引"] = pred_classes
    for i in range(10):
        print(f"预测为：{t[result_df['预测类别索引'][i]]}")
        print(f"实际为：{t[int(result_df.iloc[i,205])]}")
        image_filename = 'G:\photos of the robotic arm//'+t[result_df["预测类别索引"][i]]+'-back.jpg'  # 例如："data/hand_gesture_05.jpg"
        open_image_by_filename(image_filename)
    #result_df["预测类别名称"] = pred_class_names

    # 3. 添加每个类别的概率列（如“ClassA概率”“ClassB概率”）
    #for i in range(NUM_CLASSES):
    #    result_df[f"{class_name_mapping[i]}_概率"] = pred_probs[:, i]

    # 4. 保存结果到CSV
    result_df.to_csv("./prediction_results.csv", index=False, encoding="utf-8")
    print("预测完成！结果已保存至 prediction_results.csv")

    # 5. 打印前5条结果（快速查看）
    #print("\n前5条预测结果：")
    #print(result_df[["预测类别名称"] + [f"{class_name_mapping[i]}_概率" for i in range(NUM_CLASSES)]].head())

