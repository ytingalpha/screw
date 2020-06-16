import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import pandas as pd
import numpy as np
import math as mt

(x, y), (test_x, test_y) = datasets.cifar100.load_data()

PI = mt.pi

def rgbThsv(r,g,b):
    Cmax = max(r,g,b)
    Cmin = min(r,g,b)
    delta = Cmax - Cmin

    def getH():
        if Cmax == Cmin:
            return 0
        elif Cmax == r and g >= b:
            H = 60 * (int(g) - int(b)) / delta
            return H
        elif Cmax == r and g < b:
            H = 60 * (int(g) - int(b)) / delta + 360
            return H
        elif Cmax == g:
            bug = int(b) - int(r)
            H = 60 * bug / delta + 120
            return H
        elif Cmax == b:
            H = 60 * (int(r) - int(g)) / delta + 240
            return H
    H = int(getH())

    if Cmax == 0:
        S = 0
    else:
        S = int(delta / Cmax * 100)
    
    phi = (S * 2 + H) * mt.pi / 180

    a = mt.asinh(phi*mt.pi) / 2
    b =  mt.pow(mt.pi,2)
    c = phi * mt.pi * mt.pow(mt.pow(phi,2) * b + 1,0.5) / 2
    return a + c

# R = (99 * 360 * PI / 180 + 359 * PI / 180)
R = 468.6807592917638

TrainNum = 50000
train_x = []
train_y = y[:TrainNum]

# Get train and label data
for val in range(TrainNum):
    imgData = x[val]
    newData = []
    for i in range(32):
        for j in range(32):
            r = imgData[i][j][0]
            g = imgData[i][j][1]
            b = imgData[i][j][2]
            newData.append(rgbThsv(r,g,b))
    train_x.append(newData)

print('Train and valid data ok')

train_x = np.array(train_x)

train_x = tf.cast(train_x / R,tf.float32)
# 115.7529163

train_y = tf.one_hot(train_y,depth=100)
train_y = tf.squeeze(train_y)

# 定义神经网络
model = tf.keras.models.Sequential()
# 添加第一个全连接层
model.add(tf.keras.layers.Flatten(input_shape=(32,32)))
model.add(tf.keras.layers.Dense(
    units=512,
    kernel_initializer = 'normal',
    activation = 'relu'
))
# 添加第二个全连接层
model.add(tf.keras.layers.Dense(
    units = '256',
    kernel_initializer = 'normal',
    activation = 'relu'
))
# 输出层
model.add(tf.keras.layers.Dense(100,activation='softmax'))
# 定义训练模式
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
# 设置训练参数
train_epochs = 1000
batch_size = 50
# 训练模型
train_hisotory = model.fit(
    train_x,
    train_y,
    validation_split = 0.0125,
    epochs = train_epochs,
    batch_size = batch_size,
    verbose = 2
)