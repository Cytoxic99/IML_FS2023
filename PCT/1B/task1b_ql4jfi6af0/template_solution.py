# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from scipy.special import huber
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

def gradient_descent(X, y, w, lam, l1_ratio, alpha, l2_penalty, num_iters=1000, lr=0.01):
    
    n_samples, n_features = X.shape

    for i in range(num_iters):
        y_pred = np.dot(X, w)
        residuals = y_pred - y
        grad = 2*np.dot(X.T, residuals)/n_samples + l2_penalty*w
        grad += alpha*np.sign(w)
        
        lr_i = lr * 0.1**(i // 10)
        w -= lr_i*grad
        
        # Apply the L1/L2 regularization
        w = np.sign(w)*np.maximum(np.abs(w) - alpha*lr, 0)
        w *= 1 - lr*l2_penalty

    return w


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((X.shape[0], 21))
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    
    
    X_transformed[:, 0:5] = X_norm
    X_transformed[:, 5:10] = X_norm**2
    X_transformed[:, 10:15] = np.exp(X_norm)
    X_transformed[:, 15:20] = np.cos(X_norm)
    X_transformed[:, 20] = 1
    
    
    return X_transformed


def Lasso_reg(X, y, alpha):
    X_transformed = transform_data(X)
    clf = Lasso(alpha=alpha)
    clf.fit(X_transformed, y)
    w = clf.coef_

    return w

def fit(X, y, lam, l1_ratio = 0.5):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    
    n_samples, n_features = X_transformed.shape


    alpha = l1_ratio * lam
    l2_penalty = (1 - l1_ratio) * lam


    A = np.dot(X_transformed.transpose(), X_transformed) + l2_penalty * np.eye(n_features, dtype=np.float64)
    A_inv = np.linalg.inv(A).astype(np.float64)
    B = np.dot(X_transformed.transpose(), y)

    w = np.dot(A_inv, B)

    w = np.sign(w) * np.maximum(np.abs(w) - alpha, 0)
    # w = gradient_descent(X_transformed, y, w, lam, l1_ratio, alpha, l2_penalty)
    
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    lambdas = np.linspace(0, 1000, 10000)
    split = 5
    kf = KFold(n_splits=split, shuffle=True)
    
    cv_lam = []
    for lam in lambdas:
        lam_sum = 0
        for train, val in kf.split(X):
            X_train, X_val = X[train], X[val]
            y_train, y_val = y[train], y[val]
            
            w = fit(X_train, y_train, lam)
            
            X_val_trans = transform_data(X_val)
            y_pred = np.dot(X_val_trans, w)
            
            #lam_sum += np.mean((y_pred - y_val)**2)
            hbr = huber(1, (y_pred - y_val))
            lam_sum += np.mean(hbr)
        lam_avg = lam_sum / split
        cv_lam.append(lam_avg) 
    
    best_lam = lambdas[np.argmin(cv_lam)]
    print("Best avg: " +  str(cv_lam[np.argmin(cv_lam)]))
    print("The best lambda: " + str(best_lam))
    w = fit(X, y, best_lam, l1_ratio=0.5)   
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")