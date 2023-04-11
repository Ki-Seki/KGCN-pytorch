"""
基于用户的协同过滤推荐系统（User-based Collaborative Filtering Recommender System）
"""
from math import sqrt


def sim_distance(prefs, person1: str, person2: str):
    """
    :param prefs: 一个用户偏好字典，形如“用户-物品：评分”
    :param person1: 第一个用户
    :param person2: 第二个用户
    :return: 基于欧几里得距离的 person1 和 person2 的相似度
    """
    # 得到 person1 和 person2 共同评价过的电影列表
    si = {item for item in prefs[person1] if item in prefs[person2]}

    # 如果没有共同评价过的电影，相似度为 0
    if not si:
        return 0

    # 计算所有差值的平方和
    sum_of_squares = sum(
        pow(prefs[person1][item] - prefs[person2][item], 2)
        for item in si
    )

    return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(prefs, p1, p2):
    """
    :param prefs: 一个用户偏好字典，形如“用户-物品：评分”
    :param p1: 第一个用户
    :param p2: 第二个用户
    :return: p1 和 p2 的皮尔逊相关系数
    """
    # 得到双方都曾评价过的物品列表
    si = {item: 1 for item in prefs[p1] if item in prefs[p2]}

    # 如果没有共同之处，返回 0
    if not si:
        return 0

    # 得到列表元素个数
    n = len(si)

    # 对所有偏好求和
    sum1 = sum(prefs[p1][it] for it in si)
    sum2 = sum(prefs[p2][it] for it in si)

    # 求平方和
    sum1_sq = sum(pow(prefs[p1][it], 2) for it in si)
    sum2_sq = sum(pow(prefs[p2][it], 2) for it in si)

    # 求乘积之和
    p_sum = sum(prefs[p1][it] * prefs[p2][it] for it in si)

    # 计算皮尔逊相关系数
    num = p_sum - (sum1 * sum2 / n)
    den = sqrt((sum1_sq - pow(sum1, 2) / n) * (sum2_sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    pearson = num / den
    return pearson


def sim_cosine(prefs, p1, p2):
    """
    :param prefs: 一个用户偏好字典，形如“用户-物品：评分”
    :param p1: 第一个用户
    :param p2: 第二个用户
    :return: p1 和 p2 的余弦相似度
    """

    # 得到 p1 和 p2 共同评价过的电影列表
    si = {item for item in prefs[p1] if item in prefs[p2]}

    # 如果没有共同评价过的电影，相似度为 0
    if not si:
        return 0

    num = 0  # 余弦相似度分子部分
    norm_p1 = 0  # p1 对应向量的范数
    norm_p2 = 0  # p2 对应向量的范数
    for item in si:
        num += prefs[p1][item] * prefs[p2][item]
        norm_p1 += prefs[p1][item] * prefs[p1][item]
        norm_p2 += prefs[p2][item] * prefs[p2][item]
    norm_p1 **= 0.5
    norm_p2 **= 0.5

    cosine = num / (norm_p1 * norm_p2)
    return cosine


# 预测用户对一个未看过电影的评分
def predicate(prefs, person, movie, similarity):
    if person not in prefs:
        return 0
    if movie in prefs[person]:  # 如果看过就返回真实评价
        return prefs[person][movie]
    pred = 0
    sim_sum = 0
    for other in prefs:
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        if sim <= 0:
            continue

        if movie in prefs[other]:
            pred += sim * prefs[other][movie]
            sim_sum += sim
    if sim_sum == 0:
        return 0
    return pred / sim_sum  # 归一化


# 利用所有他人评价的加权平均，为某人提供建议
def get_recommendations(prefs, person, similarity):
    totals = {}  # total score: movie
    sim_sums = {}
    for other in prefs:
        # 不和自己作比较
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # 忽略评价值小于等于零的情况
        if sim <= 0:
            continue

        for item in prefs[other]:
            # 只对自己未看过的影片评价
            if item not in prefs[person]:  # or prefs[person][item] == 0:
                # 相似度 * 打分
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # 相似度之和
                sim_sums.setdefault(item, 0)
                sim_sums[item] += sim

    # 建立一个归一化的列表
    rankings = [(total / sim_sums[item], item) for item, total in totals.items()]

    rankings.sort(reverse=True)
    return rankings
