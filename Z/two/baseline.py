import numpy as np
from sklearn import linear_model
import torch

def baseline_linear_model(train_input, train_target, test_input, test_target):
    """
    Function to set a baseline with a simple classifier. Since we know the shape
    of the data, we augment it with squared coordinates.
    :input: train and test sets as torch tensors.
    prints the accuracy on the test set.
    """

    def add_squared(data):
        """
        append the sum of the coordinates squared to the test_input
        :input: numpy array containing the coordinates of each point
        :output: input appended with the sum of the square of the coordinates
        """
        square = np.power(data,2)[:,0]+np.power(data,2)[:,1].reshape(1,-1)
        square = np.transpose(square)
        return np.append(data,square,axis=1)

    # train the model
    X = train_input.numpy()
    X = add_squared(X)
    Y = train_target.numpy()
    clf = linear_model.SGDClassifier(max_iter=5000)
    clf.fit(X, Y.ravel())

    # test the function
    test = test_input.numpy()
    test = add_squared(test)
    Y_test = test_target.numpy()
    nb_errors = 0
    for i,x in enumerate(clf.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print('Baseline accuracy:',100-nb_errors/test.shape[0]*100,'%')
