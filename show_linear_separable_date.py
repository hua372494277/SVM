# show the linear separable date with specified
# mean and cov
import numpy as np
import matplotlib.pyplot as plt

def gen_lin_separable_data(n_samples):
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1, y1 = np.random.multivariate_normal(mean1, cov, n_samples).T
    plt.plot(X1, y1, 'rx')

    X2, y2 = np.random.multivariate_normal(mean2, cov, n_samples).T

    plt.plot(X2, y2, 'bo')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    gen_lin_separable_data(500)
