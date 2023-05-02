# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

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
    best_r2 = float('-inf')
    best_nu = 2.5
    best_ls_mat = 0
    best_alpha = 0
    best_noise = 0
    alpha_g = 10000
    '''
    ls_rbf = np.linspace(0.5, 3, 15)
    
    for ls in ls_rbf:
            kernel = RBF(length_scale=ls) + Matern(nu=best_nu) + RationalQuadratic(alpha=alpha_g) + WhiteKernel()
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
            print(f"LS_rbf: {ls}\tR2 Score: {avg_r2}")

            if avg_r2 > best_r2:
                best_kernel = kernel
                best_r2 = avg_r2
                best_ls_rbf = ls
                
    
    print("Best LS_rbf: " + str(best_ls_rbf))
    ls_mat = np.linspace(0.5, 3, 15)
    
    for ls in ls_mat:
            kernel = RBF(length_scale=best_ls_rbf) + Matern(length_scale= ls, nu=best_nu) + RationalQuadratic(alpha=alpha_g) + WhiteKernel()
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
            print(f"LS_MAT: {ls}\tR2 Score: {avg_r2}")

            if avg_r2 > best_r2:
                best_kernel = kernel
                best_r2 = avg_r2
                best_ls_mat = ls
    print("Best LS_mat: " + str(best_ls_mat))
    '''
    alphas = np.linspace(1000000, 10000000, 10)
    
    for alpha in alphas:
            kernel = RBF(length_scale=0.679) + Matern(length_scale=1.93, nu=2.5) + RationalQuadratic(alpha=1e+05, alpha_bounds=(1, alpha) ,length_scale=1) + WhiteKernel(noise_level=1)
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
            print(f"Alpha: {alpha}\tR2 Score: {avg_r2}")

            if avg_r2 > best_r2:
                best_kernel = kernel
                best_r2 = avg_r2
                best_alpha = alpha
    print("Best alpha: " + str(best_alpha))
    noises = np.linspace(0.5, 3, 15)
    '''
    for noise in noises:
            kernel = RBF(length_scale=best_ls_rbf) + Matern(nu=best_nu, length_scale=best_ls_mat) + RationalQuadratic(alpha=best_alpha) + WhiteKernel(noise_level=noise)
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
            print(f"Noise: {noise}\tR2 Score: {avg_r2}")

            if avg_r2 > best_r2:
                best_kernel = kernel
                best_r2 = avg_r2
                best_noise = noise
    print("Best noise: " + str(best_noise))
    
    best_kern = f"\nBest Kernel: {best_kernel}\tBest R2 Score: {best_r2}"
    print(best_kern)'''
    
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