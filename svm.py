'''
SVM class
'''
import numpy as np
import cvxopt 

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class SVM(object):
    """docstring for SVM"""
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        
        if C is not None:
            self.C = float(C)
        else:
            self.C = C
    

    def fit(self, features, label):
        n_samples, n_features = features.shape

        K = np.zeros( (n_samples, n_samples) )
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(features[i], features[j])

        P = cvxopt.matrix(np.outer(label, label) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1 )
        A = cvxopt.matrix(label, (1,n_samples), tc='d')
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
        self.sv = features[sv]
        self.sv_y = label[sv]

        print "%d support vectors out of %d points" % (len(self.a), n_samples)

        # Intercept
        self.b = 0
        for n in range( len(self.a) ):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])

        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None


    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b


    def predict(self, X):
        return np.sign( self.project(X))



if __name__ == "__main__":
    import pylab as pl


    def plot_2D_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        print a1
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        print b1
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("equal")
        pl.show()

    def test_2D_linear():
        X_train = np.array([[3.0, 3.0], [4.0, 3.0], [1.0, 1.0]])
        y_train = np.array([1, 1, -1])
        X_test = np.array([[3.0, 5.0], [0.0, 1.0] ])
        y_test = np.array([1, -1])

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_2D_margin(X_train[y_train==1], X_train[y_train==-1], clf)


    def plot_3D_margin(X1_train, X2_train, clf):
        # print X1_train
        # print X2_train
        # print clf.sv
        # print clf.w
        # print clf.b

        def f(x, y, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x -w[1] * y - b + c) / w[2]

        from mpl_toolkits.mplot3d import Axes3D
        fig = pl.figure()

        ax = fig.add_subplot(111, projection = '3d')
        ax.set_aspect('equal')
        # plot training dataset
        ax.scatter(X1_train[:,0], X1_train[:,1], X1_train[:,2], c="b", marker = "o")
        ax.scatter(X2_train[:,0], X2_train[:,1], X2_train[:,2], c="r", marker = "o")

        # plot hyperplane
        x = y = np.arange(0, 4, 0.1)
        X, Y = np.meshgrid(x, y)
        zs = np.array([f(x, y, clf.w, clf.b) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        pl.show()
        # pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        # pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        # pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # # w.x + b = 0
        # a0 = -4; a1 = f(a0, clf.w, clf.b)
        # print a1
        # b0 = 4; b1 = f(b0, clf.w, clf.b)
        # print b1
        # pl.plot([a0,b0], [a1,b1], "k")

        # # w.x + b = 1
        # a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        # b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        # pl.plot([a0,b0], [a1,b1], "k--")

        # # w.x + b = -1
        # a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        # b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        # pl.plot([a0,b0], [a1,b1], "k--")

        # pl.axis("equal")
        # pl.show()

    def test_3D_linear():
        X_train = np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]])
        y_train = np.array([1, -1, -1])
        X_test = np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 0.0] ])
        y_test = np.array([1, -1])

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_3D_margin(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_linear():
        return




    test_3D_linear()