from typing import Any
import urllib.request
import subprocess
import logging
import zipfile
import shutil
import random
import json
import sys
import os

from transformers import AutoTokenizer, AutoConfig
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.utils import train_test_split, batch_split, plot_curves
from core.metrics import accuracy, precision, recall, f1
from core.losses import AbstractLoss, CrossEntropyLoss
from core.optimizers import AbstractOptimizer, Adam
from core.models import AbstractModel, BERT
from core.data import Tensor
  
load_dotenv() 
BASE_PATH = f"{os.getcwd()}/lab5"
OUTPUT_DIR = f"{BASE_PATH}/data"

# ----------------------
# Get data
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

def get_pretrain_data():
    # Init data constants
    RULM_REPO_URL = "https://github.com/IlyaGusev/rulm.git"
    RULM_REPO_PATH = f"{OUTPUT_DIR}/rulm"
    MODEL_PATH = f"{OUTPUT_DIR}/rulm/models/lid.176.bin"
    WIKI_DUMP_URL = "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2"
    WIKI_DUMP_PATH = f"{OUTPUT_DIR}/ruwiki-latest-pages-articles.xml.bz2"
    WIKI_JSONL_PATH = f"{OUTPUT_DIR}/ruwiki.jsonl"

    # Clone github repo
    if not os.path.exists(RULM_REPO_PATH):
        logging.info("Cloning RULM repository...")
        subprocess.run(
            ["git", "clone", RULM_REPO_URL, RULM_REPO_PATH],
            check=True
        )
    else:
        logging.info("RULM repository already exists, skipping clone.")

    # Download dataset
    if not os.path.exists(WIKI_DUMP_PATH):
        logging.info("Downloading Wikipedia dump...")
        urllib.request.urlretrieve(WIKI_DUMP_URL, WIKI_DUMP_PATH, reporthook = progress_hook)
    else:
        logging.info("Wikipedia dump already exists, skipping download.")

    # Download model and add to repo
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            MODEL_PATH,
        )

    # Parse dataset
    if not os.path.exists(WIKI_JSONL_PATH):
        logging.info("Converting Wikipedia dump to JSONL...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "data_processing.convert_wiki",
                WIKI_DUMP_PATH,
                WIKI_JSONL_PATH,
            ],
            cwd=f"{OUTPUT_DIR}/rulm",
            check=True
        )
    else:
        logging.info("Converted JSONL already exists, skipping parsing.")

    return WIKI_JSONL_PATH

def get_finetune_data():
    # Init data constants
    TOXIC_DATASET = "aybatov/toxic-russian-comments-from-pikabu-and-2ch"
    TOXIC_ZIP_PATH = f"{OUTPUT_DIR}/toxic-russian-comments-from-pikabu-and-2ch.zip"

    # Download dataset
    if not os.path.exists(TOXIC_ZIP_PATH):
        logging.info(f"Downloading Kaggle dataset: {TOXIC_DATASET}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", TOXIC_DATASET, "-p", OUTPUT_DIR],
            check=True
        )
    else:
        logging.info("Toxic dataset already downloaded.")

    # Unzip if not already unzipped
    extracted_flag = os.path.exists(f"{OUTPUT_DIR}/russian_comments_from_2ch_pikabu.csv")
    if not extracted_flag:
        logging.info("Extracting toxic dataset ZIP...")
        with zipfile.ZipFile(TOXIC_ZIP_PATH, "r") as z:
            z.extractall(OUTPUT_DIR)
    else:
        logging.info("Toxic dataset already extracted.")

    return f"{OUTPUT_DIR}/russian_comments_from_2ch_pikabu.csv"

# ----------------------
# Split data
# ----------------------

def read_jsonl_by_rows(
    file_path: str,
    max_rows: int
) -> list[str]:
    
    # Get needed N rows from file
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            obj = json.loads(line)
            texts.append(obj["text"])

    return texts

def tokenize_input(
    texts: list[str],
    tokenizer,
    max_length: int = 128
) -> dict:
    
    # Convert text into tokens
    return tokenizer(
        texts,
        truncation=True,
        padding = "max_length",
        max_length = max_length,
        return_special_tokens_mask = True,
        return_tensors = "np"
    )

def create_mlm_inputs_and_labels(
    input_ids: np.ndarray,
    tokenizer,
    mlm_probability: float = 0.15,
    ignore_index: int = -100
)-> tuple[np.ndarray, np.ndarray]:
    
    # Set init values for input and labels
    X = input_ids.copy()
    y = input_ids.copy()
    N, L = X.shape

    # Mask special tokens
    special_tokens_mask = np.array([
        tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
        for seq in X
    ], dtype=bool)

    # Choose masked positions
    rand = np.random.rand(N, L)
    masked = (rand < mlm_probability) & ~special_tokens_mask

    # Ignore unmasked tokens
    y[~masked] = ignore_index

    # 80% [MASK]
    mask_token_mask = masked & (np.random.rand(N, L) < 0.8)
    X[mask_token_mask] = tokenizer.mask_token_id

    # 10% random token
    random_token_mask = masked & ~mask_token_mask & (np.random.rand(N, L) < 0.5)
    random_tokens = np.random.randint(0, tokenizer.vocab_size, size=(N, L))
    X[random_token_mask] = random_tokens[random_token_mask]

    return X, y

def split_pretrain_data(
    data_path: str,
    tokenizer: Any,
    test_size: float,
    max_rows: int,
    max_length: int,
    mlm_probability: float = 0.15,
    random_state: int = 42,
    ignore_index: int = -100,
    dtype: str = "fp32"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    
    # Read needed N rows from file
    texts = read_jsonl_by_rows(
        file_path = data_path,
        max_rows = max_rows
    )
    # Tokenize text
    encodings = tokenize_input(texts, tokenizer, max_length = max_length)

    # Process text into X and y
    X_np, y_np = create_mlm_inputs_and_labels(
        input_ids = encodings["input_ids"],
        tokenizer = tokenizer,
        mlm_probability = mlm_probability,
        ignore_index = ignore_index
    )
    # Convert to Tensor
    X = Tensor(X_np, dtype = dtype)
    y = Tensor(y_np, dtype = dtype)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None
    )

    return X_train, X_test, y_train, y_test

def split_finetune_data(
    data_path: str,
    test_size: float,
    random_state: int = 42,
    dtype: str = "fp32"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    # Read data
    df = pd.read_csv(data_path)
    df = df[["comment", "toxic"]]
    df["toxic"] = df["toxic"].astype(int)

    # Convert to Tensor
    X = Tensor(df["comment"].to_numpy(), dtype = dtype)
    y = Tensor(df["toxic"].to_numpy(), dtype = dtype)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size = test_size,
        random_state = random_state, 
        stratify = y
    )

    return X_train, X_test, y_train, y_test

# ----------------------
# Train/test methods
# ----------------------

def compute_metrics(
    y_pred_np: np.ndarray, 
    y_true_np: np.ndarray, 
    task: str, 
    ignore_index: int | None = None
)-> dict:
    
    metrics = {}
    if task == "mlm":
        pred_ids = np.argmax(y_pred_np, axis=-1)

        if ignore_index is not None:
            mask = y_true_np != ignore_index
            pred_ids = pred_ids[mask]
            y_true_np = y_true_np[mask]

        if len(y_true_np) == 0:
            metrics["token_acc"] = 0.0
        else:
            metrics["token_acc"] = (pred_ids == y_true_np).mean()

    elif task == "classification":
        num_cls = y_pred_np.shape[1]
        pred_cls = np.argmax(y_pred_np, axis=1)
        metrics["acc"] = accuracy(pred_cls, y_true_np, num_cls)
        metrics["prec"] = precision(pred_cls, y_true_np, num_cls)
        metrics["rec"] = recall(pred_cls, y_true_np, num_cls)
        metrics["f1"] = f1(pred_cls, y_true_np, num_cls)

    return metrics

def run_epoch(
    X: Tensor, 
    y: Tensor, 
    model: AbstractModel, 
    loss_fn: AbstractLoss, 
    optimizer: AbstractOptimizer,
    task: str,
    batch_size: int = 32,
    train: bool = True,
    device: str = "cuda:0",
    ignore_index: int | None = None
):
    stats = {}
    for Xb, yb in tqdm(batch_split(X, y, batch_size=batch_size),
                       total=(len(X) + batch_size - 1) // batch_size,
                       desc="Training batches" if train else "Validation batches"):
        # Send data to device
        Xb = Xb.to_device(device)
        yb = yb.to_device(device)

        # Do forward pass
        y_pred = model(Xb)

        # Calculate loss
        loss = loss_fn(y_pred, yb).mean()
        loss_val = loss.to_numpy()

        stats.setdefault("loss", []).append(loss_val)

        if train:
            loss_fn.backward(y_pred, yb)
            optimizer.step()
            optimizer.zero_grad()

        # Calculate metrics
        y_pred_np = y_pred.to_numpy()
        yb_np = yb.to_numpy()

        batch_metrics = compute_metrics(
            y_pred_np, yb_np, task, ignore_index
        )

        for k, v in batch_metrics.items():
            stats.setdefault(k, []).append(v)

    # Calculate mean over batches
    for k in stats:
        stats[k] = float(np.mean(stats[k]))

    return stats

def start_train_test_pipeline(
    X_train: Tensor, 
    X_test: Tensor,
    y_train: Tensor, 
    y_test: Tensor,
    model: AbstractModel,
    task: str,
    loss_fn: AbstractLoss,
    optimizer: AbstractOptimizer,
    epochs: int,
    batch_size: int,
    test_step: int,
    results_path: str,
    device: str = "cuda:0",
    ignore_index: int | None = None
):
    # set task for model
    model.set_task(task_name = task)

    train_hist = {}
    test_hist = {}
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        # train
        train_stats = run_epoch(
            X_train, 
            y_train,
            model, 
            loss_fn, 
            optimizer,
            task = task,
            batch_size = batch_size,
            train = True,
            device = device,
            ignore_index = ignore_index
        )

        for k, v in train_stats.items():
            train_hist.setdefault(k, []).append(v)

        # test
        if (epoch + 1) % test_step == 0:
            test_stats = run_epoch(
                X_test, 
                y_test,
                model, 
                loss_fn,
                optimizer = None,
                task = task,
                batch_size = batch_size,
                train = False,
                device = device,
                ignore_index = ignore_index
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
                f"{results_path}/train_{k}_{task}.png"
            )

        for k in test_hist:
            plot_curves(
                np.arange(len(test_hist[k])),
                test_hist[k],
                f"Test {k}",
                "epochs",
                k,
                f"{results_path}/test_{k}_{task}.png"
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

    # Get datasets
    pt_dataset_path = get_pretrain_data()
    ft_dataset_path = get_finetune_data()

    # Init constants
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    TOKENIZER_NAME = "cointegrated/rubert-tiny2"
    MODEL_NAME = "cointegrated/rubert-tiny2"
    N_ROWS = 10000
    MLM_PROB = 0.15
    IGNORE_INDEX = -100
    TEST_SIZE = 0.2
    TEST_STEP = 2
    DROPOUT = 0.1
    PRETRAIN_EPOCHS = 2
    FINETUNE_EPOCHS = 2
    BATCH_SIZE = 8
    LR = 1e-4
    DEVICE = "cuda:0"
    DTYPE = "fp32"

    # Load HF models
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, clean_up_tokenization_spaces = False)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Set maximum seq_length
    MAX_SEQ_LENGTH = tokenizer.model_max_length // 4

    # init parts
    model = BERT(
        vocab_size = tokenizer.vocab_size,
        d_model = config.hidden_size,
        num_heads = config.num_attention_heads,
        num_layers = config.num_hidden_layers,
        d_ff = config.intermediate_size,
        use_rope = False,
        max_seq_length = MAX_SEQ_LENGTH,
        dropout = DROPOUT
    )
    model = model.to_device(DEVICE)
    loss_fn = CrossEntropyLoss(model = model, ignore_index = IGNORE_INDEX)
    optimizer = Adam(model = model, lr = LR, reg_type = "l2")

    # Pretrain BERT
    X_train, X_test, y_train, y_test = split_pretrain_data(
        data_path = pt_dataset_path,
        tokenizer = tokenizer,
        test_size = TEST_SIZE,
        max_rows = N_ROWS,
        max_length = MAX_SEQ_LENGTH,
        mlm_probability = MLM_PROB,
        random_state = SEED,
        ignore_index = IGNORE_INDEX,
        dtype = DTYPE
    ) 
    start_train_test_pipeline(
        X_train, 
        X_test,
        y_train, 
        y_test,
        model,
        task = "mlm",
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = PRETRAIN_EPOCHS,
        batch_size = BATCH_SIZE,
        test_step = TEST_STEP,
        results_path = f"{BASE_PATH}/results",
        device = DEVICE,
        ignore_index = IGNORE_INDEX
    )
    exit()

    # Finetune BERT
    X_train, X_test, y_train, y_test = split_finetune_data(
        data_path = pt_dataset_path,
        test_size = TEST_SIZE,
        random_state = SEED,
        dtype = DTYPE
    )
    start_train_test_pipeline(
        X_train, 
        X_test,
        y_train, 
        y_test,
        model,
        task = "classification",
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = FINETUNE_EPOCHS,
        batch_size = BATCH_SIZE,
        test_step = TEST_STEP,
        results_path = f"{BASE_PATH}/results",
        device = DEVICE
    )
