from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 估计器 - 线性回归 - 正规方程
def estimator_linear_regression_api():
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 估计器预测
    print("正规方程 - 预测准确率: ", estimator.score(x_test, y_test))
    print("正规方程 - 权重系数: ", estimator.coef_, ", 偏置：", estimator.intercept_)

    # 回归评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程 - 均方误差:", error)


# 估计器 - 线性回归 - 梯度下降
def estimator_linear_sgd_api():
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器
    estimator = SGDRegressor(penalty="l1", eta0=0.05, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 估计器预测
    print("梯度下降 - 预测准确率: ", estimator.score(x_test, y_test))
    print("梯度下降 - 权重系数: ", estimator.coef_, ", 偏置：", estimator.intercept_)

    # 回归评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降 - 均方误差:", error)


# 估计器 - 线性回归 - 岭回归
def estimator_linear_ridge_api():
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器 相当于SGDRegressor(penalty="l2")
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 估计器预测
    print("岭回归 - 预测准确率: ", estimator.score(x_test, y_test))
    print("岭回归 - 权重系数: ", estimator.coef_, ", 偏置：", estimator.intercept_)

    # 回归评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归 - 均方误差:", error)


estimator_linear_regression_api()
estimator_linear_sgd_api()
estimator_linear_ridge_api()
