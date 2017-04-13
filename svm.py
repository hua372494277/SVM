'''
SVM class
'''
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class svm(object):
    """docstring for svm"""
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        
        if C is not None:
            self.C = float(C)
        else:
            self.C = C
    

    def fit(self, features, label):
        n_samples, n_features = features.shape

        K = np.zeros(n_samples. n_samples)
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1 )
        A = cvxopt.matrix(y, (1,n_samples), tc='d')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix( np.diag(np.ones(n_samples) * -1 ))
            h = cvxopt.matrix( np.zeros(n_samples) )


        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        #Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        print "%d support vectors out of %d points" % (len(self.a), n_samples)

        # Intercept
        self.b = 0
        for n in range( len(self.a) ):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind, sv])

        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None


    





