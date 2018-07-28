import numpy as np
import matplotlib.pyplot as plt

def get_y(x):
    f_x = (x + 2)**2 - 16*np.exp(-((x-2)**2))
    return f_x

x = np.arange(-8,8,0.001)
y = list(map(lambda u: get_y(u),x))
plt.plot(x,y)
# plt.show()

def get_grad(x):
    deriv = (2*x + 4)-16*(-2*x+4)*np.exp(-((x-2)**2))
    return deriv

def gradient_descent(start_x, func, grad):
    # Precision of the solution
    prec = 0.0001
    # Use a fixed small step size
    step_size = 0.1
    # max iterations
    max_iter = 100
    x_new = start_x
    res = []
    for i in range(max_iter):
        x_old = x_new
        # Use beta = -1 for gradient descent
        x_new = x_old - step_size * grad(x_new)
        f_x_new = func(x_new)
        f_x_old = func(x_old)
        res.append([x_new, f_x_new])
        if(abs(f_x_new-f_x_old)<prec):
            print("change in function values too small, leaving")
            return np.array(res)
    print("exceed maximum number of iterations, leaving")
    return np.array(res)

x_0 = -8
res = gradient_descent(x_0, get_y, get_grad)
plt.plot(res[:,0], res[:,1], '+')

x_1 = 8
res = gradient_descent(x_1, get_y, get_grad)
plt.plot(res[:,0], res[:,1], '+')


plt.show()
