import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def identify(x):
    return x

def forward(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identify(a3)
    return y

def softmax(a):
    c = np.max(a)
    # e^(a-c)乗
    exp_a = np.exp(a - c)
    # e^(a0) ~ e^(an)までの総和
    sum_exp_a = np.exp(exp_a)
    # 全体を1としたときのaの値の大きさ
    y = exp_a / sum_exp_a

    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5],
                              [0.2, 0.4, 0.6]])
    network["W2"] = ([[0.1, 0.4],
                      [0.2, 0.5],
                      [0.3, 0.6]])
    network["W3"] = np.array([[0.1, 0.3],
                              [0.2, 0.4]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["b2"] = np.array([0.1, 0.2])
    network["b3"] = np.array([0.1, 0.2])
    return network

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = le-7 # log(0)を防ぐため
    reutrn -np.sum(t * np.log(y + delta))

# f(x)の微分を求める
def numerical_diff(f,x):
    h = le-4    # Doubleの誤差の問題で、あまりに小さい値にすると丸められてしまう。
    # 1. f(x+h)-f(x)で求まる傾きは(x+h)とx間の傾きになってしまう。(本当はxの地点での傾きを求めたく、厳密にはhを0に限りなく近づける必要があるが、そうできない為)
    # 2. (x+h)と(x-h)間の傾きを求めることで、xでの傾きを少ない誤差で求められるらしい
    return (f(x+h) - f(x-h)) / (2*h) 


# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

train_size = x_train.shape[0]
batch_size = 3
batch_mask = np.random.choice(train_size, batch_size)

print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch[0].size)

y = np.array([[0, 0.2, 0.2 ,0.6, 0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.2 ,0.4, 0, 0, 0, 0, 0, 0]])
print(y.ndim)
t = np.array(np.array([0, 0.2, 0.2 ,0.6, 0, 0, 0, 0, 0, 0]))

print(np.arange(10))
# print(y[np.arange(10), y])

y = np.array([[0, 0.2, 0.2 ,0.6, 0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.2 ,0.4, 0, 0, 0, 0, 0, 0]])
print(y[[0,1], 3])

# batch_size = 1
# print(y[np.arange(batch_size), 3])
# print(y[np.arange(batch_size), t])
