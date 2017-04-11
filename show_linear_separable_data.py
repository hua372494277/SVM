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

def gen_3_lin_separable_data(n_samples):
    # generate training data in the 3-d case
    mean1 = np.array([0, 2, 2])
    mean2 = np.array([2, 0, -2])
    cov = np.array([[2, 0.6, 0.8], [0.6, 2, -0.6], [0.8, -0.6, 2]])
    x1, y1, z1 = np.random.multivariate_normal(mean1, cov, n_samples).T

    x2, y2, z2 = np.random.multivariate_normal(mean2, cov, n_samples).T

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    ax = fig.add_subplot(111, projection = '3d')

    # print x1
    # print y1
    # print z1
    ax.scatter(x1, y1, z1, c="r", marker ="o")
    ax.scatter(x2, y2, z2, c="b", marker = "x")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__ == "__main__":
    gen_3_lin_separable_data(2000)
