import urllib.request
import subprocess
import logging
import zipfile
import sys
import os

from dotenv import load_dotenv
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.utils import train_test_split, batch_split, plot_curves
from core.losses import AbstractLoss
from core.optimizers import AbstractOptimizer, Adam
from core.models import AbstractModel
from core.data import Tensor
  
load_dotenv() 
BASE_PATH = f"{os.getcwd()}/lab5"
OUTPUT_DIR = f"{BASE_PATH}/data"

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

    # Init train constants
    EMB_DIM = 128
    TEST_SIZE = 0.15
    TEST_STEP = 4
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-4
    DEVICE = "cuda:0"
    DTYPE = "fp32"