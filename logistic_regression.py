import numpy as np
class LogisticRegression(object):
    # Parameters
    #  eta           : lerning rate
    #  n_iter        : number of training times
    #  random_state  : seed
    #  cost          : cost

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []

        for _ in [0] * self.n_iter:
            liner_combination = self.linear_combination(X)
            update = self.activation(liner_combination)
            errors = y-update
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]  += self.eta * errors.sum()
            cost = -y.dot(np.log(update)) - ((1-y).dot(np.log(1-update)))
            self.cost_.append(cost)

        return self
    
    def linear_combination(self, X): # sum(w[1:]*x[1:]+w[0])
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X): # identity function
        return 1.0/(1.0+np.exp(-X))

    def predict(self, X):
        return np.where(self.activation(self.linear_combination(X)) >= 0.5, 1, 0)

