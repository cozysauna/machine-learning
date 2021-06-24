import numpy as np
# || θ.T.dot(Xi) - Yi || 
def normal_equation(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]
    best_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return best_theta