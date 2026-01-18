import numpy as np

from core.data import Tensor

def train_test_split(
    X: Tensor,
    y: Tensor,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    stratify: Tensor| None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split arrays into random train and test subsets"""

    if stratify is not None and stratify.shape[0] != X.shape[0]:
        raise ValueError("stratify must have the same length as X")

    # Get objects count
    n_samples = X.shape[0]
    # Handle test_size
    n_test = int(n_samples * test_size)
    # Set random generator
    rng = np.random.default_rng(seed=random_state)

    # Stratified split
    if stratify is not None:
        stratify_np = stratify.to_numpy()
        indices = np.arange(n_samples)

        train_indices = []
        test_indices = []

        classes, class_counts = np.unique(stratify_np, return_counts=True)

        for cls, cls_count in zip(classes, class_counts):
            cls_indices = indices[stratify_np == cls]

            if shuffle:
                rng.shuffle(cls_indices)

            cls_test_size = int(cls_count * test_size)

            test_indices.extend(cls_indices[:cls_test_size])
            train_indices.extend(cls_indices[cls_test_size:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

    # Normal split
    else:
        indices = np.arange(n_samples)

        if shuffle:
            rng.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def create_sequences(
    X: Tensor, 
    y: Tensor, 
    window_size: int = 1
)-> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create data sequnce"""
    dtype = X.dtype
    device = X.device
    seq_X, seq_y = [], []
    for i in range(len(X) - window_size):
        seq_X.append(X[i:i + window_size].to_numpy())
        seq_y.append(y[i + window_size].to_numpy())

    return Tensor(seq_X, dtype, device), Tensor(seq_y, dtype, device)

def timeseries_train_test_split(
    X: Tensor, 
    y: Tensor, 
    test_size: float = 0.2, 
    window_size: int = 1
)-> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split timeseries data into train and test subsets"""
    # Get objects count
    n_samples = X.shape[0]
    # Handle test_size
    n_test = int(n_samples * test_size)

    # Split data
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]

    # Get sequences
    X_train, y_train = create_sequences(X_train, y_train, window_size)
    X_test, y_test = create_sequences(X_test, y_test, window_size)

    return X_train, X_test, y_train, y_test 

def batch_split(
    X: Tensor, 
    y: Tensor, 
    batch_size: int = 8, 
    shuffle: bool = True, 
    random_state: int = 42
):
    """Split data into mini-batches"""
    # Get objects count
    n_samples = X.shape[0]
    # Shuffle indices
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indices)

    # Create generator
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]