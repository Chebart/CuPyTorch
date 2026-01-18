from .abstract_loss import AbstractLoss
from core.data import Tensor

class CrossEntropyLoss(AbstractLoss):
    """L(y_true, y_pred) = -sum(y_true_i * log(y_pred_i))"""

    def __init__(self, model, ignore_index = None):
        super().__init__(model)
        self.ignore_index = ignore_index

    def _convert_labels_to_one_hot(self, num_cls, labels):
        return Tensor.eye(num_cls, dtype=labels.dtype, device=labels.device)[labels]

    def forward(self, y_pred, y_true):
        # Ignore index if needed
        if self.ignore_index is not None:
            valid_mask = y_true != self.ignore_index
            y_pred = y_pred[valid_mask]
            y_true = y_true[valid_mask]

        # Return 0 if everything is ignored
        if y_true.shape[0] == 0:
            return Tensor(0.0, dtype = y_pred.dtype, device = y_pred.device)
        
        # Calculate loss
        y_true_oh = self._convert_labels_to_one_hot(y_pred.shape[1], y_true)
        loss = -(y_true_oh * (y_pred + self.eps).log()).sum() / y_true_oh.shape[0]

        return loss

    def backward(self, y_pred, y_true):
        # If ignore_index is set
        if self.ignore_index is not None:
            dLdy_full = Tensor.zeros(y_pred.shape, dtype = y_pred.dtype, device = y_pred.device)
            valid_mask = y_true != self.ignore_index
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]

            if y_true_valid.shape[0] == 0:
                return dLdy_full

            y_true_oh = self._convert_labels_to_one_hot(y_pred_valid.shape[1], y_true_valid)
            dLdy_valid = -y_true_oh / (y_pred_valid * y_true_oh.shape[0] + self.eps)
            dLdy_full[valid_mask] = dLdy_valid
            dLdy = dLdy_full
        else:
            y_true_oh = self._convert_labels_to_one_hot(y_pred.shape[1], y_true)
            dLdy = -y_true_oh / (y_pred * y_true_oh.shape[0] + self.eps)

        return self.model.backward(dLdy)
