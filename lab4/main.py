import logging
import shutil
import sys
import os

from sklearn.datasets import fetch_openml
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from core.utils import train_test_split, batch_split, plot_curves, \
                       save_restoration_grid, standard_normalization
from core.losses import AbstractLoss, MSELoss, BCELoss, L1Loss
from core.optimizers import AbstractOptimizer, Adam
from core.models import AbstractModel, VAEEncoder, Generator, Discriminator
from core.data import Tensor
  
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

if __name__ == "__main__":
    # Init constants
    Z_DIM = 128
    TEST_SIZE = 0.15
    TEST_STEP = 4
    EPOCHS = 100
    BATCH_SIZE = 64
    ENC_LR = 1e-4
    GEN_LR = 1e-4
    DISC_LR = 1e-4
    DEVICE = "cuda:0"
    DTYPE = "fp32"

    # Create needed directories
    results_path = f"{os.getcwd()}/lab4/results"
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}/gen_res", exist_ok=True)
    os.makedirs(f"{results_path}/train", exist_ok=True)
    os.makedirs(f"{results_path}/test", exist_ok=True)

    # Setup logger
    logging.basicConfig(
        filename=f"{results_path}/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get dataset
    dataset = fetch_openml("CIFAR_10", version=1)
    X = dataset.data.astype('float32') / 255
    X = standard_normalization(X, 0.5, 0.5)
    y = dataset.target.astype('int')

    # convert to numpy
    X = Tensor(X.to_numpy().reshape(-1, 3, 32, 32), dtype = DTYPE, device=DEVICE)
    y = Tensor(y.to_numpy(), dtype = DTYPE, device=DEVICE)
    # Split data on train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)
    # init parts VAEEncoder, Generator, Discriminator
    models = {
        "enc": VAEEncoder(in_channels = 3, z_dim = Z_DIM).to_device(DEVICE),
        "gen": Generator(z_dim = Z_DIM, out_channels = 3).to_device(DEVICE),
        "disc": Discriminator(in_channels = 3, out_features = 1).to_device(DEVICE)
    }
    losses = {
        "rec_loss": MSELoss(model = models["gen"]),
        #"rec_loss": L1Loss(model = models["gen"]),
        "g_loss": BCELoss(model = models["disc"]), # since we calculate g_loss grad through disc
        "d_loss": BCELoss(model = models["disc"])
    }
    optimizers = {
        "enc": Adam(model = models["enc"], lr = ENC_LR, beta1 = 0.5, beta2 = 0.999, reg_type = None),
        "gen": Adam(model = models["gen"], lr = GEN_LR, beta1 = 0.5, beta2 = 0.999, reg_type = None),
        "disc": Adam(model = models["disc"], lr = DISC_LR, beta1 = 0.5, beta2 = 0.999, reg_type = None)
    }

    # Train loop
    train_stats_by_epochs = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    test_stats_by_epochs = {"g_loss": [], "d_loss": [], "rec_loss": [], "kl_loss": []}
    for epoch in range(EPOCHS):
        logging.info(f"epoch: {epoch+1}")
        train_stats = train_fn(X_train, y_train, models, losses, optimizers)
        # Add train statistics
        for key, val_lst in train_stats.items():
            if len(val_lst) == 0: continue
            val_lst = np.array(val_lst)
            train_stats_by_epochs[key].append(np.mean(val_lst))

        # Test model each TEST_STEP iteration
        if (epoch + 1) % TEST_STEP == 0:
            test_stats = test_fn(X_test, y_test, models, losses)
            # Add test statistics
            for key, val_lst in test_stats.items():
                if len(val_lst) == 0: continue
                val_lst = np.array(val_lst)
                test_stats_by_epochs[key].append(np.mean(val_lst))

    # draw training/testing stats
    for key, val in train_stats_by_epochs.items():
        plot_curves(
            np.arange(len(val)) + 1,
            np.array(list(val)),
            f"Train {key} plot",
            "epochs",
            f"Train {key}",
            f"{results_path}/train/{key}.png"
        )
    
    for key, val in test_stats_by_epochs.items():
        plot_curves(
            np.arange(len(val)) + 1,
            np.array(list(val)),
            f"Test {key} plot",
            "epochs",
            f"Test {key}",
            f"{results_path}/test/{key}.png"
        )