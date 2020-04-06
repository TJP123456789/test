import numpy as np
from sklearn import neighbors
import pandas as pd


def knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat ** 2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=lambda s: s[0], reverse=True)

    return sortedClassCount[0][0]


if __name__ == '__main__':
    pd_data = pd.read_csv('.\花的数据集.csv')
    X = np.array(pd_data.loc[:, ('SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm')])
    y = np.array(pd_data.loc[:, 'Species'])
    X_test = []
    y_test = []
    X_data = []
    y_data = []
    for i in range(len(y)):
        if ((i + 1) % 20 == 0):
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_data.append(X[i])
            y_data.append(y[i])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # 调用函数
    for i in range(len(X_test)):
        X_predict = knn(X_test[i], X_data, y_data, 5)
        print('分类标签', X_predict, '实际标签', y_test[i])

''''# 直接调用库
    knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
    knn.fit(X_data, y_data)  # 导入数据进行训练
    y_predict = knn.predict(X_test)
    print('分类标签', y_predict,)
    print('实际标签',y_test)'''''