from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# 估计器 - 朴素贝叶斯算法
def estimator_bayes_api():
    # 获取数据
    news = fetch_20newsgroups(subset="all")
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # 提取特征值
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    print("训练集: ", x_train.shape, " 测试集: ", x_test.shape)
    # 使用朴素贝叶斯算法训练
    estimator = MultinomialNB(alpha=1.0)
    estimator.fit(x_train, y_train)

    print("预测准确率: ", estimator.score(x_test, y_test))


estimator_bayes_api()
