from .abstract_loss import AbstractLoss
from core.data import Tensor

class CrossEntropyLoss(AbstractLoss):
    """L(y_true, y_pred) = -sum(y_true_i * log(y_pred_i))"""

    def __init__(self, model, ignore_index = None):
        super().__init__(model)
        self.ignore_index = ignore_index
        self.valid_mask = None

    def forward(self, y_pred, y_true):
        # Ignore index if set
        if self.ignore_index is None:
            self.valid_mask = Tensor.ones(y_true.shape, device = y_pred.device).astype("bool")
        else:
            self.valid_mask = (y_true != self.ignore_index).astype("bool")
            
        # Use mask to get true class
        y_pred = y_pred[self.valid_mask]
        y_true = y_true[self.valid_mask]

        # Return 0 if everything is ignored
        if y_true.shape[0] == 0:
            return Tensor(0.0, dtype = y_pred.dtype, device = y_pred.device)

        # Select predicted probability of true class
        self.idx = y_true.astype("int32")
        self.batch_idx = Tensor.arange(len(self.idx), device = y_pred.device).astype("int32")
        true_probs = y_pred[self.batch_idx, self.idx]

        return -(true_probs + self.eps).log().mean()

    def backward(self, y_pred, y_true):  
        # Create full matrix
        dLdy = Tensor.zeros(y_pred.shape, dtype=y_pred.dtype, device=y_pred.device)

        # Calculate gradient
        y_pred_f = y_pred[self.valid_mask]
        B = y_pred_f.shape[0]
        grad = -1 / (y_pred_f[self.batch_idx, self.idx] * B + self.eps)
        dLdy[self.valid_mask, self.idx] = grad

        return self.model.backward(dLdy)