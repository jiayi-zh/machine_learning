import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# 归一化
def preprocessing_normalized_api():
    # 读取数据
    data = pd.read_csv("doc\\preprocessing_normalized.csv")
    print("原始数据类型: ", type(data), "原始数据: \n", data)
    # 取所有行, 取前三列
    data = data.iloc[:, :3]
    # 实例化一个转换器
    transfer = MinMaxScaler(feature_range=(1, 10))
    # 转换
    data_new = transfer.fit_transform(data)
    print("转换后数据类型: ", type(data_new), "数据: \n", data_new)


# 标准化
def preprocessing_standard_api():
    # 读取数据
    data = pd.read_csv("doc\\preprocessing_normalized.csv")
    print("原始数据类型: ", type(data), "原始数据: \n", data)
    # 取所有行, 取前三列
    data = data.iloc[:, :3]
    # 实例化一个转换器
    transfer = StandardScaler()
    # 转换
    data_new = transfer.fit_transform(data)
    print("转换后数据类型: ", type(data_new), "数据: \n", data_new)
    return None


# preprocessing_normalized()
preprocessing_standard_api()
