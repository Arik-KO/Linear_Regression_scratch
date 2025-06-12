import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiple_cost import compute_cost
from compute_gradient import compute_gradient
from gradient_descent import gradient_descent

data = pd.read_csv('house.csv')
x11 = data.x1.values
x21 = data.x2.values
x31 = data.x3.values
x41 = data.x4.values
y1 = data.y.values
x1 = x11 / np.max(x11)
x2 = x21 / np.max(x21)
x3 = x31 / np.max(x31)
x4 = x41 / np.max(x41)
y = y1 / np.max(y1)
x = np.column_stack((x1,x2,x3,x4))
print('first five elements o y_train: \n', y[:5])
print('first five elements of x_train: \n', x[:5])
w_initial = np.zeros(x.shape[1])
b_initial = 0
iterations = 10
alpha = 0.01
w, b, j_his, _ = gradient_descent(x, y, w_initial, b_initial, compute_cost, compute_gradient, alpha, iterations)
prediction = np.dot(x, w.T) + b
plt.figure(1)
plt.scatter(x11, y1, marker='x', c='r')
plt.plot(x11, prediction*np.max(y1), c='b')
plt.figure(2)
plt.scatter(x2,y,marker='x',c='y')
plt.plot(x2, prediction, c='r')
plt.show()
