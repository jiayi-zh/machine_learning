from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# 估计器 - KNN 算法
def estimator_knn_api():
    # 获取数据
    iris = load_iris()
    print("鸢尾花数据集: \n", iris.data.shape)

    # 划分数据集 -> 训练集(特征值+目标值) + 测试集(特征值+目标值)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=10)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    # 用训练集的标准对特征值做标准化
    x_test = transfer.transform(x_test)

    # KNN算法估计器
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 训练
    estimator.fit(x_train, y_train)

    # 模型评估
    # 直接比较真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("比较真实值和预测值: \n", y_test == y_predict)
    # 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)


# 估计器 - KNN 算法 - 添加网格搜索与交叉验证调优
def estimator_knn_tuning_api():
    # 获取数据
    iris = load_iris()
    print("鸢尾花数据集: \n", iris.data.shape)

    # 划分数据集 -> 训练集(特征值+目标值) + 测试集(特征值+目标值)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=5)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    # 用训练集的标准对特征值做标准化
    x_test = transfer.transform(x_test)

    # KNN算法估计器
    estimator = KNeighborsClassifier(n_neighbors=1, p=2)

    # 网格搜索与交叉验证调优
    estimator = GridSearchCV(estimator, param_grid={"n_neighbors": [1, 3, 5, 7]}, cv=10)

    # 训练
    estimator.fit(x_train, y_train)

    # 模型评估
    # 直接比较真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("比较真实值和预测值: \n", y_test == y_predict)
    # 计算准确率
    score = estimator.score(x_test, y_test)
    # 这是对于测试集来说估计器的准确率
    print("准确率为: ", score)

    # 最优结果
    print("最佳参数: ", estimator.best_params_)
    # 这是对于验证集来说估计器的准确率
    print("最佳结果: ", estimator.best_score_)
    print("最佳估计器: ", estimator.best_estimator_)
    print("交叉验证结果: ", estimator.cv_results_)


# estimator_knn_api()
estimator_knn_tuning_api()
