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
from tqdm import tqdm
import pandas as pd
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.metrics import mse, mae
from core.utils import plot_curves
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
    else:
        logging.info("dichalcogenides_public dataset already extracted")

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

def structure_to_graph(structure, target, cutoff=3.0):

    # Get coords of atoms
    coords = np.array(structure.cart_coords, dtype=np.float32)
    # Get atomic numbers used as node features
    atomic_numbers = np.array(
        [site.specie.number for site in structure],
        dtype=np.float32
    )

    # Get number of atoms
    N = len(coords)

    # Build adjacency matrix using distance cutoff
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Convert data to Tensors
    x = Tensor(atomic_numbers.reshape(-1, 1), dtype="fp32")
    adj = Tensor(adj, dtype="fp32")
    y = Tensor(np.array([target], dtype=np.float32), dtype="fp32")

    return x, adj, y

def build_graph_dataset(df, df_type: str = "train"):
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {df_type} df"):
        x, adj, y = structure_to_graph(row.structures, row.targets)
        dataset.append((x, adj, y))

    return dataset

# ----------------------
# Train/test methods
# ----------------------

def energy_within_threshold(prediction, target, threshold=0.02):
    prediction = np.array(prediction)
    target = np.array(target)

    error_energy = np.abs(target - prediction)
    success = np.sum(error_energy < threshold)

    return success / len(target)

def compute_metrics(y_true, y_pred):
    return {
        "mse": mse(y_pred, y_true),
        "mae": mae(y_pred, y_true),
        "ewt": energy_within_threshold(y_pred, y_true),
    }

def run_epoch(
    model: AbstractModel, 
    dataset: list[tuple[Tensor, Tensor, Tensor]], 
    loss_fn: AbstractLoss, 
    optimizer: AbstractOptimizer, 
    train: bool = True,
    device: str = "cuda:0" 
):

    stats = {}
    for x, adj, y in tqdm(dataset, total=len(dataset), desc="Training" if train else "Validation"):
        # Send data to device
        x = x.to_device(device)
        adj = adj.to_device(device)
        y = y.to_device(device)

        # Do forward pass
        y_pred = model(x, adj)

        # Calculate loss
        loss = loss_fn(y_pred, y).mean()
        loss_val = loss.to_numpy()

        stats.setdefault("loss", []).append(loss_val)

        if train:
            loss_fn.backward(y_pred, y)
            optimizer.step()
            optimizer.zero_grad()

        # Calculate metrics
        y_pred_np = y_pred.to_numpy()
        y_np = y.to_numpy()

        metrics = compute_metrics(y_pred_np, y_np)

        for k, v in metrics.items():
            stats.setdefault(k, []).append(v)

    # Calculate mean over dataset
    for k in stats:
        stats[k] = float(np.mean(stats[k]))

    logging.info(f"{'Train' if train else 'Test'} loss: {stats['loss']}")

    return stats

def train_test_pipeline(
    model: AbstractModel,
    train_data: list[tuple[Tensor, Tensor, Tensor]],
    test_data: list[tuple[Tensor, Tensor, Tensor]],
    loss_fn: AbstractLoss,
    optimizer: AbstractOptimizer,
    epochs: int,
    test_step: int,
    results_path: str,
    device: str = "cuda:0" 
):

    train_hist = {}
    test_hist = {}
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        # train
        train_stats = run_epoch(
            model,
            train_data,
            loss_fn, 
            optimizer,
            train = True,
            device = device,
        )
        for k, v in train_stats.items():
            train_hist.setdefault(k, []).append(v)

        # test
        if (epoch + 1) % test_step == 0:
            test_stats = run_epoch(
                model,
                test_data,
                loss_fn, 
                optimizer,
                train = False,
                device = device,
            )
            for k, v in test_stats.items():
                test_hist.setdefault(k, []).append(v)

        # draw plots
        for k in train_hist:
            plot_curves(
                np.arange(len(train_hist[k])),
                train_hist[k],
                f"Train {k}",
                "epochs",
                k,
                f"{results_path}/train_{k}.png"
            )

        for k in test_hist:
            plot_curves(
                np.arange(len(test_hist[k])),
                test_hist[k],
                f"Test {k}",
                "epochs",
                k,
                f"{results_path}/test_{k}.png"
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
    EPOCHS = 24
    LR = 1e-4
    DEVICE = "cuda:0"
    DTYPE = "fp32"

    # Get datasets
    dataset_path = download_dataset()
    train_df, test_df = prepare_dataset(dataset_path, test_size = TEST_SIZE, seed = SEED)
    train_dataset = build_graph_dataset(train_df, df_type = "train")
    test_dataset  = build_graph_dataset(test_df, df_type = "test")

    # Init parts
    model = GCN(in_features = 1, hidden_features = 32, out_features = 1)
    loss_fn = MSELoss(model = model)
    optimizer = Adam(model, lr=1e-3)

    train_test_pipeline(
        model,
        train_dataset,
        test_dataset,
        loss_fn,
        optimizer,
        epochs = EPOCHS,
        test_step = TEST_STEP,
        results_path = f"{BASE_PATH}/results",
        device = DEVICE
    )
