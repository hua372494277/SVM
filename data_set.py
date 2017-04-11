# generate positive and negative data set for training and testing
import matplotlib.pyplot as plt
import numpy as np

def gen_multivariate_normalData(mean, cov, n_samples):
    # the output of mean.shape is (nL,)
    (n_variates, ) = mean.shape
    (n_cov, m_cov) = cov.shape
    assert n_variates == n_cov, "Dimension of mean and cov Unconsistent"

    return np.random.multivariate_normal(mean, cov, n_samples)

'''
Generate the positive and negative data set
Parameter:
    n_samples = # of positive data + # of negative data
Return:
    pos_dataset
    pos_label 
    neg_dataset
    neg_label
'''
def generate_pos_neg_dataset(n_samples = 500):
    # 1 generate the positive data set
    pos_mean = np.array([0, 2])    
    pos_cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    pos_dataset = gen_multivariate_normalData(mean = pos_mean, \
                                                cov = pos_cov, \
                                                n_samples = n_samples / 2)
    pos_label = np.ones(n_samples / 2)

    # 2 generate the negative data set
    neg_mean = np.array([2, 0])    
    neg_cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    neg_dataset = gen_multivariate_normalData(mean = neg_mean, \
                                                cov = neg_cov, \
                                                n_samples = n_samples / 2)
    neg_label = np.ones(n_samples / 2) * (-1)

    return pos_dataset, pos_label, neg_dataset, neg_label

'''
Split all data into 2 sets: training set and test set
This function will split the # percentage of positive and negative dataset into training dataset
And the rest data are testset.
Parameter:
    percent: how many positive and negative dataset will be used to training
Return:
    training_dataset
    training_label
    test_dataset
    test_label
'''
def generate_train_test_dataset(pos_dataset, pos_label, neg_dataset, neg_label, percent = 0.9):

    return training_set, test_set

if __name__ == "__main__":
    pos_dataset, pos_label, neg_dataset, neg_label = generate_pos_neg_dataset(n_samples = 2000)
    # print pos_dataset
    # print pos_label

    # print neg_dataset
    # print neg_label
    # # Show the distribution of data set
    # plt.plot(pos_dataset.T[0], pos_dataset.T[1], "ro")
    # plt.plot(neg_dataset.T[0], neg_dataset.T[1], "bx")
    # plt.axis('equal')
    # plt.show()