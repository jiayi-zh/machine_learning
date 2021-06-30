import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


# 特征降维 - 过滤式 - 方差选择法
def feature_selection_variance_api():
    # 读取数据
    data = pd.read_csv("doc\\preprocessing_normalized_small.csv")
    print("原始数据类型: ", type(data), "shape:", data.shape, "原始数据: \n", data)
    # 实例化一个转换器
    transfer = VarianceThreshold(threshold=1)
    # 转换
    data_new = transfer.fit_transform(data)
    print("转换后数据类型: ", type(data_new), "shape:", data_new.shape, "数据: \n", data_new)


# 特征降维 - 过滤式 - 相关系数 - 皮尔逊相关系数
def feature_selection_pearsonr_api():
    # 读取数据
    data = pd.read_csv("doc\\preprocessing_normalized_small.csv")
    print("原始数据类型: ", type(data), "shape:", data.shape, "原始数据: \n", data)
    # 计算两个特征之间的相关性 -> 第一个返回值就是相关系数
    r = pearsonr(data["milage"], data["Liters"])
    print("相关性为: ", r)


# 主成分分析 - PCA降维
def decomposition_pca_api():
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    print("原始数据类型: ", type(data), "原始数据: \n", data)
    # 初始化一个转换器, 指定减少到2个特征
    transfer = PCA(n_components=2)
    data_new = transfer.fit_transform(data)
    print("指定减少到2个特征, 降维后数据类型: ", type(data_new), "数据: \n", data_new)
    # 初始化一个转换器, 指定保留95%特征
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    print("指定保留95%特征, 降维后数据类型: ", type(data_new), "数据: \n", data_new)


# feature_selection_variance_api()
# feature_selection_pearsonr_api()
decomposition_pca_api()
