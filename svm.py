'''
SVM class
'''

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

        



