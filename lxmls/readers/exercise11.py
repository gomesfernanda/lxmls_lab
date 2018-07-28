import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE ON THE GUIDE

a = np.arange(-5, 5, 0.1)
f_x = np.power(a,2)
plt.plot(a, f_x)

plt.xlim(-5,5)
plt.ylim(-5,15)

k = np.array([-2,0,2])
plt.plot(k, k**2, "bo")
for i in k:
    plt.plot(a, (2*i)*a-(i**2))

plt.show()
