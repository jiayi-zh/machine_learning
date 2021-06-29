import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# 标准化
def feature_selection_variance_api():
    # 读取数据
    data = pd.read_csv("doc\\preprocessing_normalized.csv")
    print("原始数据类型: ", type(data), "原始数据: \n", data)
    # 取所有行, 取前三列
    data = data.iloc[:, :3]
    # 实例化一个转换器
    transfer = VarianceThreshold()
    # 转换
    data_new = transfer.fit_transform(data)
    print("转换后数据类型: ", type(data_new), "数据: \n", data_new)
    return None


feature_selection_variance_api()
