import os
import pandas as pd
import numpy as np
import random

def process_xlsx_with_pandas(directory):
    a = []
    """
    使用pandas处理指定目录下的XLSX文件：
    1. 删除每个文件的第一行
    2. 将列名重命名为X0到X7

    参数:
        directory (str): XLSX文件所在目录路径

    返回:
        tuple: (成功处理的文件数, 处理失败的文件数)
    """
    success_count = 0
    fail_count = 0

    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return (success_count, fail_count)

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.lower().endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            a.append(filename)
            try:
                # 读取Excel文件（跳过原第一行，因为要删除）
                # header=None 表示不将任何行作为列名，后续手动设置
                df = pd.read_excel(file_path, header=None, skiprows=1)

                # 确保数据至少有8列（如果不足会用NaN填充，多余的列会被截断）
                # 保留前8列
                df = df.iloc[:, :10]
                # 填充可能的缺失列（如果原数据不足8列）
                if df.shape[1] < 10:
                    for col in range(df.shape[1], 10):
                        df[col] = pd.NA

                # 设置新列名 X0 到 X7
                df.columns = [f'X{i}' for i in range(10)]

                # 保存修改（覆盖原文件）
                df.to_excel(file_path, index=False)

                print(f"已处理文件: {filename}")
                success_count += 1

            except Exception as e:
                print(f"处理文件 '{filename}' 时出错: {str(e)}")
                fail_count += 1

    print(f"\n处理完成：成功 {success_count} 个文件，失败 {fail_count} 个文件")
    print(a)
    return (success_count, fail_count)
def compute_fourier_features(signal, sampling_rate=10):
    """
    计算信号的傅里叶变换，并提取幅度谱、功率谱和峰值频率

    参数:
        signal (array-like): 输入信号（可以是实数或复数序列）
        sampling_rate (float): 采样率（单位：Hz），默认1.0 Hz

    返回:
        tuple: (frequencies, spectrum, amplitude_spectrum, power_spectrum, peak_frequency)
            - frequencies: 频率轴数组（单位：Hz）
            - spectrum: 傅里叶变换后的频谱（复数数组）
            - amplitude_spectrum: 幅度谱（实数数组，与频率轴对应）
            - power_spectrum: 功率谱（实数数组，与频率轴对应）
            - peak_frequency: 峰值频率（功率谱最大值对应的频率，单位：Hz）
    """
    # 将输入信号转换为numpy数组
    signal = np.asarray(signal)
    n = len(signal)  # 信号长度

    if n == 0:
        raise ValueError("输入信号不能为空")

    # 计算快速傅里叶变换（FFT）
    spectrum = np.fft.fft(signal)

    # 计算频率轴（根据采样率和信号长度）
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    # 计算幅度谱（频谱的绝对值）
    amplitude_spectrum = np.abs(spectrum)

    # 计算功率谱（幅度的平方，除以信号长度以归一化）
    power_spectrum = (amplitude_spectrum ** 2) / n

    # 计算峰值频率（仅考虑正频率，避免对称重复的负频率干扰）
    positive_mask = frequencies >= 0
    positive_freqs = frequencies[positive_mask]
    positive_power = power_spectrum[positive_mask]

    if len(positive_power) == 0:
        raise ValueError("无法提取峰值频率（无有效正频率）")

    # 找到功率谱最大值对应的频率
    peak_idx = np.argmax(positive_power)
    peak_frequency = positive_freqs[peak_idx]

    return frequencies, spectrum, amplitude_spectrum/len(signal)*2, power_spectrum[frequencies >= 0], peak_frequency

def xlsx_to_csv(folder_path):
    """
    将指定目录下所有XLSX文件转换为CSV文件

    参数:
        folder_path (str): 包含XLSX文件的目录路径

    返回:
        tuple: (成功转换的文件数, 失败的文件数)
    """
    # 检查目录是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"目录不存在: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} 不是一个目录")

    success_count = 0
    fail_count = 0

    # 遍历目录中的所有文件
    for id,filename in enumerate(os.listdir(folder_path)):
        # 检查文件是否为XLSX格式
        if filename.lower().endswith('.xlsx') and not filename.startswith('~$'):
            # 构建完整的文件路径
            xlsx_path = os.path.join(folder_path, filename)

            try:
                # 读取Excel文件
                df = pd.read_excel(xlsx_path, engine='openpyxl')

                # 构建CSV文件名（替换扩展名）
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                csv_path = os.path.join(folder_path,'csv', csv_filename)
                #lable = os.path.splitext(filename)[0]
                df = process_csv(df)
                for i in range(500):
                    create_date(df,id)
                # 保存为CSV文件，使用逗号作为分隔符，保留索引
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                success_count += 1
                print(f"已成功转换: {filename} -> {csv_filename}")

            except Exception as e:
                fail_count += 1
                print(f"转换失败 {filename}: {str(e)}")

    print(f"\n转换完成 - 成功: {success_count}, 失败: {fail_count}")
    return success_count, fail_count


def process_csv(df):
    """
    读取CSV文件，处理X1,X3,X5,X7,X9列：合并列、统计非空行数、按最小行数截取

    参数:
        file_path (str): CSV文件路径

    返回:
        pandas.DataFrame: 处理后的DataFrame，包含5列且行数相等
    """

    # 定义需要处理的列
    target_columns = ['X1', 'X3', 'X5', 'X7', 'X9']

    # 检查目标列是否都存在
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件中缺少以下列: {', '.join(missing_cols)}")

    # 提取目标列
    target_df = df[target_columns].copy()

    # 统计每列的非空行数（不包含空值、NaN等）
    non_null_counts = {}
    for col in target_columns:
        # 排除空字符串和NaN
        non_null_mask = target_df[col].notna() & (target_df[col] != '')
        non_null_counts[col] = non_null_mask.sum()

    # 计算最小非空行数
    min_row_count = min(non_null_counts.values())
    if min_row_count == 0:
        raise ValueError("至少有一列没有有效数据（非空行数为0）")

    # 对每列按最小行数截取（保留前min_row_count行，超出部分截断）
    processed_df = target_df.head(min_row_count).copy()

    # 输出处理信息
    print("各列非空行数统计:")
    for col, count in non_null_counts.items():
        print(f"{col}: {count}行")
    print(f"\n最小非空行数: {min_row_count}行，已按此长度截取所有列")

    return processed_df

def create_date(df,id):
    random_float = random.uniform(0.1, 1)
    row = df.shape[0]
    intever = row*round(random_float, 2)
    begin = random.randint(0,row-int(intever))
    end = begin + int(intever)
    df = df.iloc[begin:end, :]
    df1 = chuli(df["X1"])
    df2 = chuli(df["X3"])
    df3 = chuli(df["X5"])
    df4 = chuli(df["X7"])
    df5 = chuli(df["X9"])
    Y = np.array([id])
    X = np.hstack([df1,df2,df3,df4,df5,Y]).reshape(1,-1)
    # 保存路径
    csv_file = r"./result.csv"

    # 追加保存
    save_to_csv_append(X, csv_file)

def chuli(df):
  f,spec,a,b,c =  compute_fourier_features(df,sampling_rate=int(0.07*len(df)))
  a = a[:20]
  b = b[:20]
  c = np.array([c])
  merged = np.hstack([a, b, c])
  return merged


def save_to_csv_append(X, csv_path):
    """
    将行向量X追加保存到CSV文件

    参数:
        X (ndarray): 待保存的行向量（形状为(1, n)）
        csv_path (str): CSV文件路径
    """
    # 检查X是否为行向量（确保是二维数组，单一行）
    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError("X必须是形状为(1, n)的行向量")

    with open(csv_path, 'a') as f:
        np.savetxt(f, X, delimiter=',', fmt='%.5f')


def one_hot_encode_numeric_category(csv_path, output_path):
    """
    读取CSV文件，将最后一列（数值型类别）转为字符串后进行独热编码，保存结果

    参数:
        csv_path (str): 输入CSV文件路径
        output_path (str): 输出CSV文件路径
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"读取CSV文件失败: {str(e)}")

    if df.empty:
        raise ValueError("CSV文件为空，无法处理")

    # 获取最后一列列名
    last_col = df.columns[-1]
    print(f"处理最后一列 '{last_col}'（数值型类别）...")

    # 关键步骤：将数值型类别转换为字符串类型（避免被误判为连续值）
    # 若原列有缺失值，可先处理（如df[last_col] = df[last_col].fillna('missing').astype(str)）
    df[last_col] = df[last_col].astype(str)

    # 执行独热编码
    df_encoded = pd.get_dummies(df, columns=[last_col], drop_first=False)

    # 保存结果
    df_encoded.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"独热编码完成，结果已保存至: {output_path}")
# 使用示例
if __name__ == "__main__":
    target_directory = r"G:\原始数据\信号值"  # 替换为你的XLSX文件目录
    #process_xlsx_with_pandas(target_directory)

    title = np.arange(206).reshape(1, -1)
    with open(r'./result.csv', 'w') as f:
        np.savetxt(f, title, delimiter=',', fmt='%.5f')
    xlsx_to_csv(target_directory)

    #one_hot_encode_numeric_category(r'./result.csv','./result_new.csv')






