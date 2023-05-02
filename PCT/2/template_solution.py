# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from scipy.stats import zscore

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation, and removes the outlier rows from the training data

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))
    
    train_df = train_df.dropna(subset=['price_CHF'])
    print(train_df)

    # Remove outlier rows
    z_scores = np.abs(zscore(train_df['price_CHF']))
    threshold = 3
    train_df = train_df[z_scores <= threshold]

    X_train = train_df.drop(['price_CHF'],axis=1)
    y_train = train_df['price_CHF']
    X_test = test_df

    X_train = X_train.iloc[:, 1:]
    X_test = X_test.iloc[:, 1:]


    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test




def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_pred: array of floats: dim = (100,), predictions on test set
    """
    #Best Kernel: RBF(length_scale=0.679) + Matern(length_scale=1.93, nu=2.5) + RationalQuadratic(alpha=1e+05, length_scale=1) + WhiteKernel(noise_level=1)  Best R2 Score: 0.9557069522518644

    best_kernel = None
    kernel = RBF(length_scale=0.679) + Matern(length_scale=1.93, nu=2.5) + RationalQuadratic(alpha=1e+05, length_scale=1) + WhiteKernel(noise_level=1)
    r2_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]


        gpr = GaussianProcessRegressor(kernel=kernel).fit(X_train_fold, y_train_fold)
        y_pred_fold = gpr.predict(X_val_fold, return_std=False)

        r2_score_fold = r2_score(y_val_fold, y_pred_fold)
        r2_scores.append(r2_score_fold)

    avg_r2 = np.mean(r2_scores)
    print(f"R2 Score: {avg_r2}")

    
    
    best_kernel = RBF(length_scale=0.679) + Matern(length_scale=1.93, nu=2.5) + RationalQuadratic(alpha=1e+05, length_scale=1) + WhiteKernel(noise_level=1)
    gpr_best = GaussianProcessRegressor(kernel=best_kernel).fit(X_train, y_train)
    y_pred = gpr_best.predict(X_test, return_std=False)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred




# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")