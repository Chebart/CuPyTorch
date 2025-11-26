from .abstract_loss import AbstractLoss

class L1Loss(AbstractLoss):
    """L(y_true, y_pred) = mean(|y_true - y_pred|)"""

    def forward(self, y_pred, y_true):
        return (abs(y_pred - y_true)).sum() / y_true.shape[0]

    def backward(self, y_pred, y_true):
        dLdy = (((y_pred - y_true) > 0) - ((y_pred - y_true) < 0)) / y_true.shape[0]
        dLdy = self.model.backward(dLdy)
        return dLdy