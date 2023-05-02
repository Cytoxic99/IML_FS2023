# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from scipy.special import huber
from sklearn.model_selection import KFold



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
    X_transformed[:, 0:5] = X
    X_transformed[:, 5:10] = X**2
    X_transformed[:, 10:15] = np.exp(X)
    X_transformed[:, 15:20] = np.cos(X)
    X_transformed[:, 20] = 1
    
    
    return X_transformed


def adam_optimizer(X, y, w, iterations, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    t = 0
    X = transform_data(X)
    for epoch in range(iterations):
        t += 1
        grad = -2*np.dot(X.T, (y - np.dot(X, w)))
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        m_hat = m/(1-beta1**t)
        v_hat = v/(1-beta2**t)
        w -= alpha*m_hat/(np.sqrt(v_hat) + epsilon)
        y_pred = np.dot(X, w)
        loss = np.divide(np.sum((y - y_pred)**2, axis=0), X.shape[0])
        print(f'Epoch {epoch}, loss is {loss}')
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
    print("The best lambda: " + str(best_lam))
    w = fit(X, y, best_lam, l1_ratio=0.5)
    w = adam_optimizer(X, y, w, iterations= 1000000)
       
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
