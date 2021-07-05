from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 估计器 - 分类 - KMeans(未完成)
def estimator_linear_logistic_api():
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器 相当于
    estimator = KMeans()
    estimator.fit(x_train, y_train)

    # 估计器预测
    print("KMeans - 预测准确率: ", estimator.score(x_test, y_test))

    # 轮廓系数
    y_predict = estimator.predict(x_test)
    score = silhouette_score(x_train, y_predict)
    print("KMeans - 轮廓系数:", score)


estimator_linear_logistic_api()
