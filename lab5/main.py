from typing import Any
import urllib.request
import subprocess
import logging
import zipfile
import random
import json
import sys
import os

from transformers import AutoTokenizer, AutoConfig
from dotenv import load_dotenv
import pandas as pd
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.utils import train_test_split, batch_split, plot_curves
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
        urllib.request.urlretrieve(WIKI_DUMP_URL, WIKI_DUMP_PATH)
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
    dtype: str = "fp32",
    device: str = "cuda:0"
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
        mlm_probability = mlm_probability
    )
    # Convert to Tensor
    X = Tensor(X_np, dtype = dtype, device = device)
    y = Tensor(y_np, dtype = dtype, device = device)

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
    dtype: str = "fp32",
    device: str = "cuda:0"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    # Read data
    df = pd.read_csv(data_path)
    df = df[["comment", "toxic"]]
    df["toxic"] = df["toxic"].astype(int)

    # Convert to Tensor
    X = Tensor(df["comment"].to_numpy(), dtype = dtype, device = device)
    y = Tensor(df["toxic"].to_numpy(), dtype = dtype, device = device)

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

def train_fn(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    models: dict[AbstractModel],
    losses: dict[AbstractLoss],
    optimizers: dict[AbstractOptimizer]
)-> np.ndarray:
    train_stats = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    for train_Xb, _ in batch_split(X_train, y_train, batch_size = BATCH_SIZE):
        # -------------------------
        # Train Discriminator
        # -------------------------
        real_preds = models["disc"](train_Xb)
        real_targets = Tensor.ones(real_preds.shape, dtype = real_preds.dtype, device = real_preds.device)
        loss_D_real = losses["d_loss"](real_preds, real_targets).mean().to_numpy()
        losses["d_loss"].backward(real_preds, real_targets)

        _, _, z = models["enc"](train_Xb)
        fake = models["gen"](z)
        fake_preds = models["disc"](fake)
        fake_targets = Tensor.zeros(fake_preds.shape, dtype = fake_preds.dtype, device = fake_preds.device)
        loss_D_fake = losses["d_loss"](fake_preds, fake_targets).mean().to_numpy()
        losses["d_loss"].backward(fake_preds, fake_targets)

        loss_D = loss_D_real + loss_D_fake
        train_stats["d_loss"].append(loss_D)
        optimizers["disc"].step()
        optimizers["disc"].zero_grad()

        # -------------------------
        # Train Encoder + Generator
        # -------------------------
        mu, logvar, z = models["enc"](train_Xb)
        recon = models["gen"](z)

        # Calculate reconstruction loss
        rec_loss = losses["rec_loss"](recon, train_Xb).mean().to_numpy()
        train_stats["rec_loss"].append(rec_loss)
        dLdz_rec = losses["rec_loss"].backward(recon, train_Xb) # get rec_loss derivative from generator

        # Calculate generator loss
        preds = models["disc"](recon)
        targets = Tensor.ones(preds.shape, dtype = preds.dtype, device = preds.device)
        loss_G = losses["g_loss"](preds, targets).mean().to_numpy()
        train_stats["g_loss"].append(loss_G)
        dLdx_fake = losses["g_loss"].backward(preds, targets) # get g_loss derivative through discriminator
        dLdz_gen = models["gen"].backward(dLdx_fake) # get g_loss derivative from generator

        # Calculate KL loss
        kl_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean().to_numpy()
        train_stats["kl_loss"].append(kl_loss)

        # Use combined derivative for encoder
        norm = mu.shape[0] * mu.shape[1]
        dKL_dmu = mu / norm
        dKL_dlogvar = 0.5 * (logvar.exp() - 1) / norm
        dLdz_gen_dmu = dLdz_gen
        dLdz_gen_dlogvar = dLdz_gen * models["enc"].eps * 0.5 * models["enc"].std
        dLdz_rec_dmu = dLdz_rec
        dLdz_rec_dlogvar = dLdz_rec * models["enc"].eps * 0.5 * models["enc"].std

        dmu_total = dLdz_gen_dmu + dKL_dmu + dLdz_rec_dmu
        dsigma_total = dLdz_gen_dlogvar + dKL_dlogvar + dLdz_rec_dlogvar
        d_encoder_out = Tensor.concat(
            [dmu_total, dsigma_total], 
            axis = 1, 
            dtype = dmu_total.dtype, 
            device = dmu_total.device
        )
        models["enc"].backward(d_encoder_out)

        optimizers["enc"].step()
        optimizers["gen"].step()
        optimizers["enc"].zero_grad()
        optimizers["gen"].zero_grad()
        optimizers["disc"].zero_grad()

    logging.info(f"train g_loss: {np.array(train_stats['g_loss']).mean()}")
    logging.info(f"train d_loss: {np.array(train_stats['d_loss']).mean()}")
    logging.info(f"train rec_loss: {np.array(train_stats['rec_loss']).mean()}")
    logging.info(f"train kl_loss: {np.array(train_stats['kl_loss']).mean()}")

    return train_stats

def test_fn(
    X_test: np.ndarray, 
    y_test: np.ndarray,
    models: list[AbstractModel],
    losses: list[AbstractLoss]
)-> np.ndarray:
    test_stats = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    for batch_idx, (test_Xb, _) in enumerate(batch_split(X_test, y_test, batch_size = BATCH_SIZE)): 
        # -------------------------
        # Test Discriminator
        # -------------------------
        real_preds = models["disc"](test_Xb)
        real_targets = Tensor.ones(real_preds.shape, dtype = real_preds.dtype, device = real_preds.device)
        loss_D_real = losses["d_loss"](real_preds, real_targets).mean().to_numpy()

        _, _, z = models["enc"](test_Xb)
        fake = models["gen"](z)
        fake_preds = models["disc"](fake)
        fake_targets = Tensor.zeros(fake_preds.shape, dtype = fake_preds.dtype, device = fake_preds.device)
        loss_D_fake = losses["d_loss"](fake_preds, fake_targets).mean().to_numpy()

        loss_D = loss_D_real + loss_D_fake
        test_stats["d_loss"].append(loss_D)

        # -------------------------
        # Test Encoder + Generator
        # -------------------------
        mu, logvar, z = models["enc"](test_Xb)
        recon = models["gen"](z)

        # Calculate losses
        rec_loss = losses["rec_loss"](recon, test_Xb).mean().to_numpy()
        test_stats["rec_loss"].append(rec_loss)

        kl_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean().to_numpy()
        test_stats["kl_loss"].append(kl_loss)

        preds = models["disc"](recon)
        targets = Tensor.ones(preds.shape, dtype = preds.dtype, device = preds.device)
        loss_G = losses["g_loss"](preds, targets).mean().to_numpy()
        test_stats["g_loss"].append(loss_G)

        # -------------------------
        # Save restored images
        # -------------------------
        recon = recon.to_numpy()
        test_Xb  = test_Xb.to_numpy()
        save_restoration_grid(recon, test_Xb, save_path = f"{results_path}/gen_res/test_batch_{batch_idx}.png")

    logging.info(f"test g_loss: {np.array(test_stats['g_loss']).mean()}")
    logging.info(f"test d_loss: {np.array(test_stats['d_loss']).mean()}")
    logging.info(f"test rec_loss: {np.array(test_stats['rec_loss']).mean()}")
    logging.info(f"test kl_loss: {np.array(test_stats['kl_loss']).mean()}")

    return test_stats

def train_test_pipeline():
    ...

if __name__ == "__main__":
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
    N_ROWS = 1000
    MLM_PROB = 0.15

    TEST_SIZE = 0.2
    TEST_STEP = 2
    DROPOUT = 0.1
    PRETRAIN_EPOCHS = 20
    FINETUNE_EPOCHS = 10
    BATCH_SIZE = 64
    LR = 1e-4
    DEVICE = "cuda:0"
    DTYPE = "fp32"

    # Load HF models
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # init parts
    model = BERT(
        vocab_size = tokenizer.vocab_size,
        d_model = config.hidden_size,
        num_heads = config.num_attention_heads,
        num_layers = config.num_hidden_layers,
        d_ff = config.intermediate_size,
        use_rope = False,
        max_seq_length = tokenizer.model_max_length,
        dropout = DROPOUT
    ).to_device(DEVICE)
    loss_fn = CrossEntropyLoss(model = model)
    optimizer = Adam(model = model, lr = LR, reg_type = "l2")

    # Get finetune data
    X_train, X_test, y_train, y_test = split_pretrain_data(
        data_path = pt_dataset_path,
        tokenizer = tokenizer,
        test_size = TEST_SIZE,
        max_rows = N_ROWS,
        max_length = tokenizer.model_max_length,
        mlm_probability = MLM_PROB,
        random_state = SEED,
        dtype = DTYPE,
        device = DEVICE
    )