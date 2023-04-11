"""
在数据集上进行的仿真实验
"""
import random
import itertools
from sklearn.metrics import roc_auc_score
import UCFRS


def get_all_movies(path='./ml-latest-small'):
    movies = {}
    is_first_line = True
    for line in open(path):
        # 跳过首行
        if is_first_line:
            is_first_line = False
            continue
        (_, id) = line.split('\t')[0:2]
        movies[id] = id
    return movies


def get_all_ratings(path='./ml-latest-small'):
    ratings = []
    is_first_line = True
    for line in open(path):
        # 跳过首行
        if is_first_line:
            is_first_line = False
            continue
        (user, item, rating) = line.strip().split('\t')[0:3]
        rating = 0 if float(rating) < 3.0 else 1
        ratings.append([user, item, rating])
    return ratings


def split_raintgs_into_train_and_test(ratings, train_ratio, randseed):
    # TODO: spliting should be corresponding to KGCN
    random.seed(randseed)
    random.shuffle(ratings)
    train_size = int(train_ratio * len(ratings))
    train = ratings[:train_size]
    test = ratings[train_size:]
    return train, test


def rating2prefs(ratings, movies):
    prefs = {}
    for rating in ratings:
        (userId, movieId, rating) = rating
        prefs.setdefault(userId, {})  # 如果还没有 userId 的评分，创建空字典；如果 userId 已经有值了，就不改变这个字典
        prefs[userId][movies[movieId]] = float(rating)
    return prefs


def experiment(seed, train_ratio, similarity):
    movies = get_all_movies('../data/product/ratings.csv')
    ratings = get_all_ratings('../data/product/ratings.csv')
    train_ratings, test_ratings = split_raintgs_into_train_and_test(ratings, train_ratio, seed)
    train_prefs = rating2prefs(train_ratings, movies)

    real_ratings = []
    predicated_ratings = []
    for rating in test_ratings:
        user = rating[0]
        movieId = rating[1]
        real = rating[2]
        pred = UCFRS.predicate(train_prefs, user, movies[movieId], similarity)
        real_ratings.append(real)
        predicated_ratings.append(pred)
    auc = roc_auc_score(real_ratings, predicated_ratings)

    return auc, real_ratings, predicated_ratings


if __name__ == '__main__':
    for seed in [1, 999, 345, 124, 3345]:
        auc, _, _ = experiment(seed, 0.8, UCFRS.sim_distance)
        print(f'{seed=}, {auc=}')
