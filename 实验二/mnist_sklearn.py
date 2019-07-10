import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def test_clf(clf, train_data, train_target, test_data, test_target):
    """Train classifier by train dada and return the accuracy of classifier in test data.
    Parameters:
    clf : object
        Classifier
        e.g.: DecisionTreeClassifier, LogisticRegression, etc.
    train_data : array, shape = [n_samples, n_features]
    train_target : array, shape = [n_samples] or [n_samples, n_outputs]
    test_data : array, shape = [n_samples, n_features]
    test_target : array, shape = [n_samples] or [n_samples, n_outputs]

    Return:
    acc : float, the accuracy of trained classifier in test data.
    """
    clf.fit(train_data, train_target)
    pred = clf.predict(test_data)
    acc = np.sum(pred==test_target) / test_data.shape[0]
    return acc

mnist = fetch_mldata('MNIST Original', data_home='./')
mnist_num = mnist.data.shape[0]
half_num = mnist_num // 2
max_pixel = np.max(mnist.data)

#pixel feature
mnist_pixel = mnist.data
#gray histogram feature
mnist_gray_hist = np.array(
    [np.bincount(pixel, minlength=max_pixel+1).tolist() for pixel in mnist_pixel]
)

shuffle_index = np.array(range(mnist_num))
np.random.shuffle(shuffle_index)

mnist_pixel_shuffle = mnist_pixel[shuffle_index]
mnist_gray_hist_shuffle = mnist_gray_hist[shuffle_index]
mnist_tar_shuffle = mnist.target[shuffle_index]

clf = DecisionTreeClassifier(random_state=0)
DecisionTree_acc_pixel = test_clf(
    clf, 
    train_data = mnist_pixel_shuffle[:half_num], 
    train_target = mnist_tar_shuffle[:half_num],
    test_data = mnist_pixel_shuffle[half_num:], 
    test_target = mnist_tar_shuffle[half_num:], 
)

DecisionTree_acc_gray_hist = test_clf(
    clf, 
    train_data = mnist_gray_hist_shuffle[:half_num], 
    train_target = mnist_tar_shuffle[:half_num],
    test_data = mnist_gray_hist_shuffle[half_num:], 
    test_target = mnist_tar_shuffle[half_num:], 
)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

Logistic_acc_pixel = test_clf(
    clf, 
    train_data = mnist_pixel_shuffle[:half_num], 
    train_target = mnist_tar_shuffle[:half_num],
    test_data = mnist_pixel_shuffle[half_num:], 
    test_target = mnist_tar_shuffle[half_num:], 
)

Logistic_acc_gray_hist = test_clf(
    clf, 
    train_data = mnist_gray_hist_shuffle[:half_num], 
    train_target = mnist_tar_shuffle[:half_num],
    test_data = mnist_gray_hist_shuffle[half_num:], 
    test_target = mnist_tar_shuffle[half_num:], 
)

clf = LinearSVC(random_state=0)

LinearSVM_acc_pixel = test_clf(
    clf, 
    train_data = mnist_pixel_shuffle[:half_num], 
    train_target = mnist_tar_shuffle[:half_num],
    test_data = mnist_pixel_shuffle[half_num:], 
    test_target = mnist_tar_shuffle[half_num:], 
)

LinearSVM_acc_gray_hist = test_clf(
    clf, 
    train_data = mnist_gray_hist_shuffle[:half_num], 
    train_target = mnist_tar_shuffle[:half_num],
    test_data = mnist_gray_hist_shuffle[half_num:], 
    test_target = mnist_tar_shuffle[half_num:], 
)

print(
    'Pixel Accuracy:\nDecisionTree: {:.2f} LogisticRegression: {:.2f} LinearSVM: {:.2f}'.format(
        DecisionTree_acc_pixel*100, Logistic_acc_pixel*100, LinearSVM_acc_pixel*100)
)
print(
    'Gray Histogram Accuracy:\nDecisionTree: {:.2f} LogisticRegression: {:.2f} LinearSVM: {:.2f}'.format(
        DecisionTree_acc_gray_hist*100, Logistic_acc_gray_hist*100, LinearSVM_acc_gray_hist*100)
)
# Pixel Accuracy:
# DecisionTree: 85.91 LogisticRegression: 91.65 LinearSVM: 87.07
# Gray Histogram Accuracy:
# DecisionTree: 24.71 LogisticRegression: 22.15 LinearSVM: 19.27