import math


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    j_history = []
    w_history = []
    for i in range(num_iters):
        dw, db = gradient_function(x, y, w_in, b_in)
        w_in = w_in - alpha*dw
        b_in = b_in - alpha*db
        if i < 10000:
            cost = cost_function(x, y, w_in, b_in)
            j_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w_in, b_in, j_history, w_history

