import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot


# TODO: need to verify

class TwoStepLDATree(BaseEstimator, ClassifierMixin):
    # Emulates decision tree, but each step is an LDA. Assumes binary (e.g. labels are only 0 or 1) classification.
    def __init__(self):
        self.lda1 = LinearDiscriminantAnalysis()
        self.lda2 = LinearDiscriminantAnalysis()
        self.lda1_bad_class = None

    def fit(self, X, y):
        self.lda1.fit(X, y)
        y_pred_1 = self.lda1.predict(X)
        missed_preds = y_pred_1 - y
        if np.sum(missed_preds == 1) > np.sum(missed_preds == -1):
            self.lda1_bad_class = 1
        else:
            self.lda1_bad_class = 0

        self.lda2.fit(
            X[np.where(y_pred_1 == self.lda1_bad_class)],
            y[y_pred_1 == self.lda1_bad_class],
        )
        return self

    def predict(self, X):
        y_pred_1 = self.lda1.predict(X)
        y_pred_2 = self.lda2.predict(X[np.where(y_pred_1 == self.lda1_bad_class)])
        y_pred = y_pred_1.copy()
        y_pred[y_pred_1 == self.lda1_bad_class] = y_pred_2
        return y_pred

    def predict_proba(self, X):
        proba_1 = self.lda1.predict_proba(X)
        preds_1 = self.lda1.predict(X)
        proba = proba_1.copy()

        proba_2 = self.lda2.predict_proba(X[np.where(preds_1 == self.lda1_bad_class)])
        proba[np.where(preds_1 == self.lda1_bad_class), :] = proba_2
        return proba

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane. Follows decision tree logic, as in, takes confidence of LDA2 if LDA1 predicts the bad class.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.
        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        check_is_fitted(self.lda1)
        check_is_fitted(self.lda2)
        # xp, _ = get_namespace(X)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        preds_1 = self.lda1.predict(X)
        scores_1 = (
            safe_sparse_dot(X, self.lda1.coef_.T, dense_output=True)
            + self.lda1.intercept_
        )

        scores_2 = (
            safe_sparse_dot(
                X[np.where(preds_1 == self.lda1_bad_class)],
                self.lda2.coef_.T,
                dense_output=True,
            )
            + self.lda2.intercept_
        )

        scores = scores_1.copy()
        scores[np.where(preds_1 == self.lda1_bad_class)] = scores_2
        # return xp.reshape(scores, -1) if scores.shape[1] == 1 else scores
        return scores
