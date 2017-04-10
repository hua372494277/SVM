# Draw random samples from a multivariate normal distribution

import numpy as np
import matplotlib.pyplot as plt

def formatOutputMultivariateNormal():
    matrix = np.random.multivariate_normal(mean, cov, 10)
    print str(matrix)
    print "\n"
    #.T is to transpose the matrix
    print str(matrix.T)
    return

if __name__ == "__main__":
    mean = [0, 0]
    cov = [ [1, 0], [0, 1]]

    formatOutputMultivariateNormal()

    