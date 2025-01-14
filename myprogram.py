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

# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
img = img.reshape(28, 28)

img_show(img)
