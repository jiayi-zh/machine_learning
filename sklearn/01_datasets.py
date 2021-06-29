from sklearn.datasets import load_iris


def sklearn_iris_datasets_api():
    # 获取鸢尾花数据集 -> 返回值 datasets.base.Bench(字典格式)
    iris = load_iris()
    # 查看鸢尾花数据集
    print("鸢尾花数据集: \n", iris)

    print("鸢尾花特征值: \n", iris["data"])
    print("鸢尾花目标值: \n", iris.target)
    print("鸢尾花特征的名字: \n", iris.feature_names)
    print("鸢尾花目标值的名字: \n", iris.target_names)
    print("鸢尾花描述: \n", iris.DESCR)


sklearn_iris_datasets_api()
