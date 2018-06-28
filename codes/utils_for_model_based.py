import random
import numpy as np
import os
import pandas as pd
import sys


def load_data(path='../data/ml-100k/'):
    """
    Descriptions: To get dataset of train, test and movies.
    Params:
        path: path of movie lens dataset
    """
    movies = {}
    for line in open(path + '/u.item', encoding='latin-1'):
        id, title = line.split('|')[0:2]
        movies[id] = title
    prefs = {}
    for line in open(path + '/u.data', encoding='latin-1'):
        user, movieid, rating, ts = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs


def split_data(data_shuffled, idx_test=0, valid_split=0.2):
    test = []
    train = []
    len_test = np.ceil(data_shuffled.shape[0]*valid_split).astype(np.int)
    test = data_shuffled[idx_test*len_test:(idx_test+1)*len_test]
    train = np.vstack(
        (
            data_shuffled[:idx_test*len_test],
            data_shuffled[min((idx_test+1)*len_test, data_shuffled.shape[0]-1):]
        )
    )
    return train, test


def gen_train_test_data(path_data):
    data = pd.read_csv(path_data, index_col=None, header=None)
    data = data.values
    data_shuffled = np.random.permutation(data)
    data_train, data_test = split_data(data_shuffled)
    # train_data
    train_data = np.zeros((944, 1683), dtype=np.int)
    for data in data_train:
        data = data[0].split('\t')
        user = int(data[0])
        movie = int(data[1])
        train_data[user][movie] = int(data[2])
    # test_data
    test_data = np.zeros((944, 1683), dtype=np.int)
    for data in data_test:
        data = data[0].split('\t')
        user = int(data[0])
        movie = int(data[1])
        test_data[user][movie] = int(data[2])
    return train_data, test_data

def sgd(data_matrix, k, alpha, lam, max_cycles):
    """使用梯度下降法进行矩阵分解。

    Args:
    - data_matrix: mat, 用户物品矩阵
    - k: int, 分解矩阵的参数
    - alpha: float, 学习率
    - lam: float, 正则化参数
    - max_cycles: int, 最大迭代次数

    Returns:
    p,q: mat, 分解后的矩阵
    """
    m, n = np.shape(data_matrix)
    # initiate p & q
    p = np.mat(np.random.random((m, k)))
    q = np.mat(np.random.random((k, n)))

    # start training
    for step in range(max_cycles):
        for i in range(m):
            for j in range(n):
                if data_matrix[i, j] > 0:
                    error = data_matrix[i, j]
                    for r in range(k):
                        error = error - p[i, r] * q[r, j]
                    for r in range(k):
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - lam * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - lam * q[r, j])

        loss = 0.0
        for i in range(m):
            for j in range(n):
                if data_matrix[i, j] > 0:
                    error = 0.0
                    for r in range(k):
                        error = error + p[i, r] * q[r, j]
                    # calculate loss function
                    loss = (data_matrix[i, j] - error) * (data_matrix[i, j] - error)
                    for r in range(k):
                        loss = loss + lam * (p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2
        print("loss:", loss.shape, loss)
        if loss < 0.001:
            break
        if step % 1000 == 0:
            print("\titer: %d, loss: %f" % (step, loss))
    return p, q

def train_by_GD(data_train, k, alpha, lam, max_cycles):
    # initialization
    m, n = np.shape(data_train)
    p = np.random.random((m, k))
    q = np.random.random((k, n))
    clock_lines = ['-', '\\', '|', '/']

    # training
    for step in range(max_cycles):
        for i in range(m):
            for j in range(n):
                if data_train[i, j] > 0:
                    error = data_train[i, j]
                    error = error - np.sum(p[i, :] * q[:, j])
                    for r in range(k):
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - lam * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - lam * q[r, j])

        loss = 0.0
        for i in range(m):
            for j in range(n):
                if data_train[i, j] > 0:
                    error = np.squeeze(np.matmul(p[i, :].reshape(1, -1), q[:, j].reshape(-1, 1)))
                    # print(p[i, :].reshape(1, -1).shape, q[:, j].reshape(-1, 1).shape, end='--')
                    # print('error.shape={}'.format(error.shape), end='--')
                    # calculate loss function
                    loss = (data_train[i, j] - error) * (data_train[i, j] - error)
                    # print(loss.shape, end=',     ')
                    # for r in range(k):
                    #     loss = loss + lam * (p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2
                    loss = loss + lam * (np.sum(np.square(p[i, :])) + np.sum(np.square(q[:, j]))) / 2

        if loss < 0.001:
            break

        sys.stdout.flush()
        sys.stdout.write(
            "iter: {}, loss: {} {}\r".format(
                step, loss, clock_lines[step%len(clock_lines)]
            )
        )
    return p, q


def main():
    data = load_data('../data/ml-100k/')
    data_shuffled = np.random.permutation(data)
    data_train, data_test = split_data(data_shuffled)
    print("data_train.shape, data_test.shape:", data_train.shape, data_test.shape)


if __name__ == '__main__':
    main()
