'''
该文档实现的功能是BP网络，利用一个样例来更新每个参数，还未仔细研究，有空的时候可以研究一下
实现的功能是异或
https://www.cnblogs.com/xuhongbin/p/6666826.html
'''
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) * np.tanh(x)

# sigmod函数
def logistic(x):
    return 1 / (1 + np.exp(-x))

# sigmod函数的导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# 输入参数layer是一个list，表示每层多少个节点和一共多少层
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        # 随机产生权重值
        self.weights = []
        for i in range(1, len(layers) - 1):  # 不算输入层，循环
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
            # print self.weights

    def fit(self, x, y, learning_rate=0.2, epochs=10000):
        x = np.atleast_2d(x)
        temp = np.ones([x.shape[0], x.shape[1] + 1])
        temp[:, 0:-1] = x
        x = temp# 变换输入X
        y = np.array(y)

        for k in range(epochs):  # 循环epochs次
            i = np.random.randint(x.shape[0])  # 随机产生一个数，对应行号，即数据集编号
            a = [x[i]]  # 抽出这行的数据集

            # 迭代将输出数据更新在a的最后一行
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            # 减去最后更新的数据，得到误差
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # 求梯度
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            # 反向排序
            deltas.reverse()

            # 梯度下降法更新权值
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a



nn = NeuralNetwork([2, 2, 1], 'tanh')
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(x, y)
# for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
#     print(i, nn.predict(i))
