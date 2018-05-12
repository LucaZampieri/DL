import numpy as np
from sklearn import linear_model
import torch

def baseline_linear_model(train_input, train_target, test_input, test_target):
    def add_squared(data):
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
    print(nb_errors/test.shape[0]*100,'%')
    