from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# 估计器 - 决策树算法
def estimator_decision_tree_api():
    # 获取数据
    iris = load_iris()

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

    # 决策树训练
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=None)
    estimator.fit(x_train, y_train)

    # 决策树估计
    print("预测准确率: ", estimator.score(x_test, y_test))

    # 可视化
    export_graphviz(estimator, "./tree.dot", feature_names=iris.feature_names)


estimator_decision_tree_api()

"""
tree.dot 输出：

digraph Tree {
node [shape=box] ;
0 [label="petal length (cm) <= 2.45\nentropy = 1.584\nsamples = 112\nvalue = [37, 39, 36]"] ;
1 [label="entropy = 0.0\nsamples = 37\nvalue = [37, 0, 0]"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
2 [label="petal width (cm) <= 1.75\nentropy = 0.999\nsamples = 75\nvalue = [0, 39, 36]"] ;
0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
3 [label="petal length (cm) <= 4.95\nentropy = 0.281\nsamples = 41\nvalue = [0, 39, 2]"] ;
2 -> 3 ;
4 [label="entropy = 0.0\nsamples = 38\nvalue = [0, 38, 0]"] ;
3 -> 4 ;
5 [label="sepal width (cm) <= 2.9\nentropy = 0.918\nsamples = 3\nvalue = [0, 1, 2]"] ;
3 -> 5 ;
6 [label="entropy = 0.0\nsamples = 2\nvalue = [0, 0, 2]"] ;
5 -> 6 ;
7 [label="entropy = 0.0\nsamples = 1\nvalue = [0, 1, 0]"] ;
5 -> 7 ;
8 [label="entropy = 0.0\nsamples = 34\nvalue = [0, 0, 34]"] ;
2 -> 8 ;
}
"""
