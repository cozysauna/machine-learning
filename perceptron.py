import numpy as np
from copy import copy
class Perceptron(object):
    # Parameters
    #  eta           : lerning rate
    #  n_iter        : number of training times
    #  random_state  : seed
    #  errors_       : errors of every iteration
    #  w_transition_  : how weight has changed

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # Initialize
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = 1 + X.shape[1]) # w_[0]:= Bias Unit
        self.w_transition_ = []
        self.errors_ = []
        for _ in [0] * self.n_iter:
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update * xi #w[1] += update*X[1], w[2] += update*X[2], ...
                self.w_[0]  += update
                if update != 0.0:
                    errors += 1
                    self.w_transition_.append(copy(self.w_))

            self.errors_.append(errors)

        return self 

    def linear_combination(self, X): # sum(w[1:]*x[1:]+w[0])
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # Dividing the data into 1 and -1, you can change the last 0 below to -1 and the threshold to 0
        return np.where(self.linear_combination(X) >= 0.5, 1, 0)
