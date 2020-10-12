import sys, os #???
sys.path.append(os.pardir) #???
import numpy as np
from common.layers import * #
from common.gradient import numerical_gradient #
from collections import OrderedDict #順番付きdict

def softmax(x):
    ma = np.max(x):
    exp_mx = np.exp(x-ma)
    sum_exp_mx = np.sum(exp_mx)
    y = exp_mx/sum_exp_mx
    return y

def cross_entropy_error(y,t):
    if y.ndim==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
#Relu classの作成

class Relu:
    def __init__(self):
        self.mask = none
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask]=0
    def backward(self, dout):
        dout[self.mask]=0
        return dout

#Affine classの作成

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out
    def backward(self,dout)
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.np.sum(dout,axis=0)
        return dx
    
#Softmax classの作成
class Softmax:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx

#TwoLayerNet classの作成(predict, loss, accuracy, numerical_gradient, gradient)
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01): #weight_init_std?
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = Softmax()
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x) #layersの各dictをつかって次に進める
        return x
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t) #教師データtと画像データxから計算した結果データyを比較してloss算出
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1) #yは各データxから計算したyの最頻値のindex、つまりどの数字と判断したか
        if t.ndim!=1: #dimension
            t=np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0]) #一致した個数/ケース数
        return accuracy
    def numerical_gradient(self,x,t): #learning時の微小変化量
        loss_W = lambda W: self.loss(x,t)　#def loss_W(W)と同義
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads
    def gradient(self,x,t):
        self.loss(x,t) #forward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values()) #['Affine1','Relu1','Affine2']
        layers.reverse() #['Affine2','Relu1','Affine1']
        for layer in layers:
            dout = layer.backward(dout)
        grads={}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
        
