import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#Load Data
data = pd.read_csv('D:\Learning\machine_learning\Hoc_May_CK\Hoc_May_CK\data.csv')
x = data.loc[:,['IoU','x0','y0','w0','h0','xb','yb','wb','hb']]
y = data.loc[:,['select']]
#chia tap train , test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(x_train.shape,y_train.shape)
#Chuẩn hóa dữ liệu
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
# from sklearn.preprocessing import StandardScaler
# std = StandardScaler()
# y_train_scaled = std.fit_transform(y_train)
# y_test_scaled = std.fit_transform(y_test)
y_train_scaled = y_train
y_test_scaled = y_test
# x_max = x_train.values.max(axis = 0, keepdims = True)
# x_min = x_train.values.min(axis = 0, keepdims = True)
# x_train_scaled2 = (x_train-x_min)/(x_max-x_min)
# y_std = y_train.values.std(axis = 0,keepdims = True)
#y_mean  = y_train.values.mean(axis = 0, keepdims = True)
# y_train_scaled2 = (y_train-y_mean)/y_std
# print(y_train_scaled)
# print(y_train_scaled2)   
#Hàm sigmoid







g = lambda z :np.exp(z) / (1+np.exp(z))
#Y dự đoán
def predict(x,w):
    z = np.dot(x,w)
    return g(z)

def predict_(x,w):
    y_pre = predict(x,w)
    y_pre[y_pre >= 0.5] = 1
    y_pre[y_pre < 0.5] = 0
    return y_pre
#Hàm Loss
def loss(x,y,w):
    y_pred = predict(x,w)
    dy = y*np.log(y_pred)+(1-y)*np.log(1-y_pred+0.000001)
    return -np.mean(dy,axis=0,keepdims= False)

#Gradient
def gradient(x,y,w):
    y_pred = predict(x,w)
    dy = y_pred-y
    dw = np.dot(x.T,dy)
    return dw

def gradiednDescent(x,y,lr,epochs):
    history = []
    #Hệ số bias
    w = np.r_[np.zeros((1,1)),np.ones((x.shape[1],1))]
    x = np.c_[np.ones((x.shape[0],1)),x]
    for i in range(epochs):
        dw = gradient(x,y,w)
        w = w-lr*dw
        l = loss(x,y,w)
        history.append(l)
    return history,w
history,w = gradiednDescent(x_train_scaled,y_train_scaled,0.001,1000)
x = np.c_[np.ones((x_test.shape[0],1)),x_test]
print(np.dot(x,w).shape)
print(y_test.shape)
plt.plot(history)
plt.title('Lịch sử độ lỗi')
plt.xlabel('Epoch')
plt.ylabel('Độ lỗi')
plt.show()
