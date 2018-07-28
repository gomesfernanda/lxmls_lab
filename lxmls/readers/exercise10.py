import numpy as np
import matplotlib.pyplot as plt
import galton as galton

galton_data = galton.load()

# Exercise 1: What are the mean height and standard deviation of all the people in the sample? What is the mean height of the fathers and of the sons?

dset_mean = galton_data.mean()
print("mean:", dset_mean)

dset_sdev = galton_data.std()
print("stardard deviation:", dset_sdev)

# Exercise 2: Plot a histogram of all the heights (you might want to use the plt.hist function and the ravel method on arrays).

plt.hist(galton_data)
plt.show()

# Exercise 3: Plot the height of the father versus the height of the son.

plt.scatter(galton_data[:,0], galton_data[:,1])
plt.show()

# Exercise 4: You should notice that there are several points that are exactly the same (e.g., there are 21 pairs with the values 68.5 and 70.2). Use the ? command in ipython to read the documentation for the numpy.random.randn function and add random jitter (i.e., move the point a little bit) to the points before displaying them. Does your impression of the data change?

galton_data_noise = galton_data + 0.5*np.random.randn(len(galton_data), 2)
plt.scatter(galton_data_noise[:,0], galton_data_noise[:,1])
plt.show()
