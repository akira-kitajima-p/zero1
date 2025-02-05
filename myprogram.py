import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def identify(x):
    return x

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

def cross_entropy_error(y,t):
    delta = 1e-7 # log(0)を防ぐため
    return -np.sum(t * np.log(y + delta))

# f(x)の微分を求める
def numerical_diff(f,x):
    h = 1e-4    # Doubleの誤差の問題で、あまりに小さい値にすると丸められてしまう。
    # 1. f(x+h)-f(x)で求まる傾きは(x+h)とx間の傾きになってしまう。(本当はxの地点での傾きを求めたく、厳密にはhを0に限りなく近づける必要があるが、そうできない為)
    # 2. (x+h)と(x-h)間の傾きを求めることで、xでの傾きを少ない誤差で求められるらしい
    return (f(x+h) - f(x-h)) / (2*h) 


# 関数fの重み配列がxに格納されており、x[idx]の偏微分をgrad[idx]に格納して返す
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    # flags=['multi_index']を渡すことで、xのインデックスをタプル形式で順番に返すイテレータを生成できる
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()
        
    return grad

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 2×3の中間層を生成する(ガウス分布で初期化)
    
    def predict(self, x):
        return np.dot(x, self.W) # xは2個の入力, 中間層との積を出力して3個の要素を得る
    
    # 損失を求める。0に近いほど損失が少ない
    def loss(self, x, t):
        z = self.predict(x)     # xに対する予想結果
        y = softmax(z)          # 結果の総和を1にする
        loss = cross_entropy_error(y, t) # 正解データtをもとに損失を求める
        return loss

net = simpleNet()

x = np.array([0.6, 0.9])
p = net.predict(x)
print(softmax(p))

t = np.array([0,0,1]) # 正解ラベル
print(net.loss(x, t))

# 正解ラベルtに対する損失の勾配(≒重みwの偏微分の結果)
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
