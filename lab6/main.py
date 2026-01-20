import subprocess
import logging
import tarfile
import shutil
import random
import json
import sys
import os

from sklearn.model_selection import train_test_split
from pymatgen.core import Structure
from pathlib import Path
import pandas as pd
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.metrics import mse
from core.utils import batch_split, plot_curves
from core.losses import AbstractLoss, MSELoss
from core.optimizers import AbstractOptimizer, Adam
from core.models import AbstractModel, GCN
from core.data import Tensor

BASE_PATH = f"{os.getcwd()}/lab6"
OUTPUT_DIR = f"{BASE_PATH}/data"

# ----------------------
# Process data
# ----------------------

def progress_hook(
    block_num: int,
    block_size: int,
    total_size: int,
):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(
            f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB "
            f"({percent:.2f}%)"
        )
        sys.stdout.flush()

def download_dataset():
    DATASET_REPO_URL = "https://github.com/HSE-LAMBDA/IDAO-2022"
    DATASET_REPO_PATH = f"{OUTPUT_DIR}/IDAO-2022"
    ARCHIVE_PATH = f"{DATASET_REPO_PATH}/data/dichalcogenides_public.tar.gz"

    # Clone github repo
    if not os.path.exists(DATASET_REPO_PATH):
        logging.info("Cloning IDAO-2022 repository...")
        subprocess.run(
            ["git", "clone", DATASET_REPO_URL, DATASET_REPO_PATH],
            check=True
        )
    else:
        logging.info("IDAO-2022 repository already exists, skipping clone.")

    extract_path = os.path.join(OUTPUT_DIR, "dichalcogenides_public")
    if not os.path.exists(extract_path):
        logging.info("Extracting dataset...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(OUTPUT_DIR)

    return extract_path

def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)

def prepare_dataset(dataset_path, test_size: float, seed: int):
    dataset_path = Path(dataset_path)

    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)

    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

    data = pd.DataFrame(
        columns=["structures"],
        index=struct.keys()
    )

    data = data.assign(
        structures=struct.values(),
        targets=targets
    )

    return train_test_split(data, test_size = test_size, random_state = seed)

# ----------------------
# Train/test methods
# ----------------------

def epoch_run(model, dataset, loss_fn, optimizer=None, train=True):
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, adj, y in dataset:
        # Forward
        node_out = model.forward(x, adj)

        # Graph-level pooling (mean)
        pred = node_out.mean(axis=0)

        loss = loss_fn(pred, y)
        total_loss += loss.item()

        if train:
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()
            optimizer.zero_grad()

        y_true.append(y.item())
        y_pred.append(pred.item())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(dataset)

    return metrics

def train_test_pipeline(
    model,
    train_data,
    test_data,
    loss_fn,
    optimizer,
    epochs=50
):
    for epoch in range(1, epochs + 1):
        train_metrics = epoch_run(
            model, train_data, loss_fn, optimizer, train=True
        )

        test_metrics = epoch_run(
            model, test_data, loss_fn, optimizer=None, train=False
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_metrics['mse']:.6f} | "
            f"Test MSE: {test_metrics['mse']:.6f}"
        )

if __name__ == "__main__":
    # Delete result directory
    if os.path.exists(f"{BASE_PATH}/results"):
        shutil.rmtree(f"{BASE_PATH}/results")

    # Create needed directories
    directories = ["data", "results"]
    for directory in directories:
        if not os.path.exists(f"{BASE_PATH}/{directory}"):
            os.makedirs(f"{BASE_PATH}/{directory}")

    # Setup logger
    logging.basicConfig(
        filename=f"{BASE_PATH}/results/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Init constants
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    TEST_SIZE = 0.2
    TEST_STEP = 2
    EPOCHS = 20
    BATCH_SIZE = 16
    LR = 1e-4
    DEVICE = "cpu"
    DTYPE = "fp32"

    # Get datasets
    dataset_path = download_dataset()
    train_df, test_df = prepare_dataset(dataset_path, test_size = TEST_SIZE, seed = SEED)
    print(train_df.shape, test_df.shape)
    exit()

    # Init parts
    model = GCN(in_features = 1, hidden_features = 32, out_features = 1)
    loss_fn = MSELoss(model = model)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train_test_pipeline(
        model,
        train_df,
        test_df,
        loss_fn,
        optimizer,
        epochs = EPOCHS
    )
