项目介绍：


数据获得：
安装：
1.运行代码的基础环境要求：最低要求：Python 3.6  推荐版本：Python 3.7+
2.核心依赖库：
pandas​ >= 1.3.0 （用于Excel/CSV文件读写、数据处理）
numpy​ >= 1.21.0（数值计算、傅里叶变换等核心运算）
openpyxl​ >= 3.0.0（隐式依赖）（pandas读取xlsx文件需要）
PyTorch​ >= 1.12.0【神经网络框架（需与torchvision版本匹配）】
torchvision​ >= 0.13.0（图像处理和预训练模型支持）
scikit-learn​ >= 1.1.0【评估指标（accuracy_score等）】
matplotlib​ >= 3.5.0（训练过程可视化）
Pillow​ >= 9.0.0（图片显示功能）等等。
3.版本兼容性注意事项：PyTorch与CUDA版本匹配；依赖冲突预防，建议使用虚拟环境（venv/conda）。

具体操作：

若想快速的获取最终结果，可跳过1.数据处理步骤，直接使用我们提供的数据集resultce.csv文件，运行train.py和Model Invocation.py

步骤：
1.数据预处理：首先运行preprocess.py文件。根据实际情况调整代码285行的原始数据目录，和222行的处理后数据保存目录。
首先，它会读取Excel文件，删掉第一行标题，把前八列改名为X0到X7。接着，处理CSV时只保留X1、X3、X5、X7、X9这几列，把有数据的行截断到一样长，去掉空值太多的列。然后对每段信号做傅里叶分析，算出幅度、能量和峰值频率这些特征，再混上随机截取的子样本数据。
最后给每个样本加个ID标签，把所有特征拼起来存成CSV。输出的结果大概是200行数据，每行有200多个特征，包括傅里叶分析的细节、原始信号统计值和类别编码。
因Excel文件有24组，故最终输出一个大约4801行数据的.csv文件。
【注意事项】
如果遇到 PermissionError，检查文件是否被其他程序占用
若 compute_fourier_features报错，确保输入信号长度 > 0
根据需求调整 process_xlsx_with_pandas中的参数（如采样率 sampling_rate）

2.运行train.py文件：
数据准备
将best_model.pth文件和train.py文件放在同一目录下，将train.py代码中的第336行中的.csv文件名改为1步骤处理后的.csv文件（或直接使用已给的数据集new_result.csv）

调整以下参数：
config = {
    'batch_size': 32,         # 批次大小
    'val_split': 0.2,         # 验证集比例
    'hidden_dims': [256,128,64,32],  # 隐藏层维度
    'lr': 0.01,               # 学习率
    'weight_decay': 1e-5,     # L2正则化系数
    'epochs': 100,            # 训练轮数
    'save_path': './best_model.pth',  # 模型保存路径
}

运行程序，训练100轮（可根据需要调整）

输出结果：
训练过程曲线图 training_curve.png
最佳模型权重 best_model.pth
验证集评估报告 classification_report

【注意事项】
根据硬件调整 batch_size（显存不足时减小）
若 CUDA out of memory，尝试减小 batch_size或使用 --no-cuda
调整 hidden_dims和 learning_rate优化模型性能


3.运行Model Invocation.py文件：
准备数据
确保代码158行的所用文件为new_result.csv
将训练好的模型权重文件best_model.pth和Model Invocation.py文件放在同一目录下
根据实际情况修改Model Invocation.py文件中214行代码中的机械手照片文件目录

修改配置参数
在 load_trained_model()中调整以下参数：
MODEL_PATH = "./best_model.pth"  # 模型权重路径
NUM_CLASSES = 24                # 类别数量

运行预测

# 在命令行中进入代码所在目录
cd /path/to/Model Invocation

# 执行预测脚本
Model Invocation.py

输出结果
预测结果保存为 prediction_results.csv
自动弹出预测类别对应的图片（需确保图片路径正确）

【注意事项】
若图片无法打开，检查文件路径是否包含中文或特殊字符
调整 predict_transform中的预处理参数（如标准化均值/标准差）




