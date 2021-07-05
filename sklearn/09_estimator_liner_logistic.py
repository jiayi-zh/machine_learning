from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 估计器 - 分类 - 逻辑回归(未完成)
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
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 估计器预测
    print("逻辑回归 - 预测准确率: ", estimator.score(x_test, y_test))
    print("逻辑回归 - 权重系数: ", estimator.coef_, ", 偏置：", estimator.intercept_)

    # 回归评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("逻辑回归 - 均方误差:", error)


estimator_linear_logistic_api()
