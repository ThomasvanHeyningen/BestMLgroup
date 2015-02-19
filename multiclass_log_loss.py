import numpy as np

class MulticlassLogLoss:


    def __init__(self):
        self.eps=1e-15
        #totally empty

    def calculate_log_loss(self, y_true, y_pred):
        """Multi class version of Logarithmic Loss metric.
        https://www.kaggle.com/wiki/MultiClassLogLoss

        Parameters
        ----------
        y_true : array, shape = [n_samples]
                true class, intergers in [0, n_classes - 1)
        y_pred : array, shape = [n_samples, n_classes]

        Returns
        -------
        loss : float
        """
        predictions = np.clip(y_pred, self.eps, 1 - self.eps)

        # normalize row sums to 1
        predictions /= predictions.sum(axis=1)[:, np.newaxis]

        actual = np.zeros(y_pred.shape)
        n_samples = actual.shape[0]
        actual[np.arange(n_samples), y_true.astype(int)] = 1
        vectsum = np.sum(actual * np.log(predictions))
        loss = -1.0 / n_samples * vectsum
        return loss

