import numpy as np
from sklearn.svm import SVC

class SVMClassifier:
    """
    基于特征的 SVM 手势识别模型

    输入:
        X: shape = (N, C, F)
        y: shape = (N,)
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ):
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma
        )

    def fit(self, X, y):
        """
        训练 SVM

        Parameters
        ----------
        X : np.ndarray, shape (N, C, F)
        y : np.ndarray, shape (N,)
        """
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)

    def predict(self, X):
        """
        预测手势类别

        Parameters
        ----------
        X : np.ndarray, shape (N, C, F)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
