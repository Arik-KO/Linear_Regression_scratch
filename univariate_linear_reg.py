import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gradient_descent import gradient_descent
from cost_linear import compute_cost
from compute_gradient import compute_gradient

data = pd.read_csv('House_Price.csv')
x_train = np.array(data.room_num.values)
y_train = np.array(data.price.values)
print('type of x_train', type(x_train))
print('first five elements of x_train: \n', x_train[:5])
print('type of y_train', type(y_train))
print('first five elements o y_train: \n', y_train[:5])
plt.figure(1)
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title('population vs profits per city')
plt.xlabel('population in city in 10000')
plt.ylabel('profits in city per $10000')

w_initial = 0
b_initial = 0
iterations = 10000
alpha = 0.03
w, b, j_his, _ = gradient_descent(x_train, y_train, w_initial, b_initial, compute_cost, compute_gradient, alpha, iterations)
prediction = np.dot(w, x_train) + b
plt.figure(2)
plt.scatter(x_train, y_train, marker='o', c='r')
plt.plot(x_train, prediction, c='b')
plt.title('population vs profits per city')
plt.xlabel('population in city in 10000')
plt.ylabel('profits in city per $10000')
plt.figure(3)
program_iterations = np.arange(10000)
plt.plot(program_iterations, j_his, c='y')
plt.xlabel('number of iterations')
plt.ylabel('Cost function')
plt.title('Iterations vs Cost_Function')
print(w)
print(b)
plt.show()

