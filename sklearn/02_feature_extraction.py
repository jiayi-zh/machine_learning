import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# 特征提取 - 字典
def feature_extraction_dict_api():
    data = [{"city": "北京", "rank": 1}, {"city": "上海", "rank": 2}, {"city": "深圳", "rank": 3}]
    # 实例化一个转换器类
    # sparse: 控制返回的sparse矩阵是否包含无特征的数据, True 不返回, False 返回
    transfer = DictVectorizer(sparse=True)
    # 将字典数据 转换成 一个sparse矩阵
    data_new = transfer.fit_transform(data)
    print("特征名称: \n", transfer.get_feature_names())
    print("提取结果: \n", data_new)
    # 将sparse矩阵 还原成 字典数据
    data_inverse = transfer.inverse_transform(data_new)
    print("还原结果: \n", data_inverse)


# 特征提取 - 文本 - 个数
def feature_extraction_text_api():
    strList = ["life is short, i like python", "life is too long, i dislike python in my life"]
    # 实例化一个转换器
    transfer = CountVectorizer(stop_words=["is"])
    # 特征提取
    data_new = transfer.fit_transform(strList)
    print("特征名称: \n", transfer.get_feature_names())
    print("提取结果类型: ", type(data_new), "提取结果: \n", data_new.toarray())


# 特征提取 - 文本 - 个数 - 中文处理
def feature_extraction_text_chinese_api():
    textList = ["这是一篇关于IT的文章",
                "这是一篇关于金融的文章"]
    # 先使用 jieba 分词器对中文分词
    newTextList = []
    for text in textList:
        newTextList.append(cut_word(text))
    # 实例化一个转换器
    transfer = CountVectorizer()
    # 特征提取
    data_new = transfer.fit_transform(newTextList)
    print("特征名称: \n", transfer.get_feature_names())
    print("提取结果类型: ", type(data_new), "提取结果: \n", data_new.toarray())


def cut_word(text):
    return " ".join(list(jieba.cut(text)))


# 特征提取 - 文本 - 个数 - TF-IDF
def feature_extraction_text_tfidf_api():
    strList = ["life is short, i like python", "life is too long, i dislike python in my life"]
    # 实例化一个转换器
    transfer = TfidfVectorizer()
    # 特征提取
    transfer.fit_transform(strList)

    data_new = transfer.fit_transform(strList)
    print("特征名称: \n", transfer.get_feature_names())
    print("提取结果类型: ", type(data_new), "提取结果: \n", data_new.toarray())


# feature_extraction_dict_api()
# feature_extraction_text_api()
# feature_extraction_text_chinese_api()
feature_extraction_text_tfidf_api()
