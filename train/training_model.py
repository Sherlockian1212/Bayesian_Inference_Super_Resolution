import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.data_loader import data_prepairing_pipeline as dataset
from utils.helper import unwrap_generator_output
from model.build_model import ModelSR

from utils.config import (
    LEARNING_RATE,
    EPOCH,
    PRETRAIN_EPOCH,
    LOSS_LOG,
    EARLY_STOPPING,
    CHANNELS,
    DOWN_SCALE_RATE,
    VERSION,
    DATA,
    DATA_LOG,
    BETA_KL,
    BETA_NLL
)

from utils.metrics import psnr, ssim
from utils.loss import SSIMLoss, CharbonnierLoss
from torchinfo import summary

MODEL_SAVE_PATH = f"../weights/best_SR_x{DOWN_SCALE_RATE}_{VERSION}_{DATA}.pth"


# ================= CHECKPOINT =================
def save_checkpoint(epoch, model, optimizer, best_val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_loss = checkpoint["best_val_loss"]
    start_epoch = checkpoint["epoch"] + 1

    print(f"🔄 Resumed training from epoch {start_epoch}")
    return start_epoch, best_val_loss


# ================= KL SUM =================
def total_kl_loss(module):
    kl = 0.0
    for m in module.modules():
        if hasattr(m, "kl_loss"):
            kl += m.kl_loss()
    return kl


# ================= MAIN =================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Device:", device)

    # ===== DATA =====
    train_loader, val_loader, test_loader = dataset(
        dataset=DATA,
        log_file=DATA_LOG
    )

    # ===== MODEL =====
    model = ModelSR(c=CHANNELS, h=128, w=128).to(device)

    try:
        summary(model, input_size=(1, CHANNELS, 128, 128), device=device.type)
    except Exception as e:
        print("⚠️ torchinfo summary failed:", e)

    # ===== LOSS =====
    criterion_charb = CharbonnierLoss()
    criterion_ssim = SSIMLoss()
    criterion_nll = nn.GaussianNLLLoss(reduction='mean', full=True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.5
    )

    # ===== RESUME =====
    checkpoint_path = MODEL_SAVE_PATH.replace(".pth", "_checkpoint.pth")

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
    else:
        best_val_loss = float("inf")

    # ===== LOG =====
    if start_epoch == 0:
        with open(LOSS_LOG, "w") as f:
            f.write("Epoch,Train_Loss,Val_Loss,Val_PSNR,Val_SSIM,Avg_KL\n")

    dataset_size = len(train_loader.dataset)
    counter = 0

    # ================= TRAIN =================
    for epoch in range(start_epoch, EPOCH):

        model.train()

        running_loss = 0.0
        running_kl = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCH}] Training")

        for lr_img, hr_img in pbar:

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            batch_size = lr_img.size(0)
            total_samples += batch_size

            kl_unscaled = total_kl_loss(model)
            kl_term = kl_unscaled / dataset_size * batch_size

            optimizer.zero_grad()

            if epoch < PRETRAIN_EPOCH:
                out = model(lr_img, sample=False)
            else:
                out = model(lr_img, sample=True)

            sr_img, mu, sigma = unwrap_generator_output(out)

            charb_loss = criterion_charb(sr_img, hr_img)
            ssim_loss = criterion_ssim(sr_img, hr_img)
            recon_loss = 0.8 * charb_loss + 0.2 * ssim_loss

            if epoch < PRETRAIN_EPOCH:
                loss = recon_loss
            else:
                nll_loss = criterion_nll(sr_img, hr_img, sigma ** 2)
                loss = recon_loss + BETA_NLL * nll_loss + BETA_KL * kl_term

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            running_kl += kl_unscaled.item()

            avg_loss = running_loss / total_samples
            avg_kl = running_kl / total_samples

            pbar.set_postfix({
                "Loss": f"{avg_loss:.6f}",
                "KL": f"{avg_kl:.6f}"
            })

        train_loss = running_loss / dataset_size
        avg_epoch_kl = running_kl / dataset_size

        # ================= VALIDATION =================
        model.eval()

        val_loss = 0
        val_psnr = 0
        val_ssim = 0

        with torch.no_grad():
            for lr_img, hr_img in val_loader:

                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)

                sr_img, _, _ = unwrap_generator_output(model(lr_img))

                charb_loss = criterion_charb(sr_img, hr_img)
                ssim_loss = criterion_ssim(sr_img, hr_img)
                recon_loss = 0.8 * charb_loss + 0.2 * ssim_loss

                val_loss += recon_loss.item() * lr_img.size(0)
                val_psnr += psnr(sr_img, hr_img) * lr_img.size(0)
                val_ssim += ssim(sr_img, hr_img).item() * lr_img.size(0)

        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        print(
            f"📉 Epoch {epoch+1}: "
            f"Train={train_loss:.6f} | "
            f"Val={val_loss:.6f} | "
            f"PSNR={val_psnr:.4f} | "
            f"SSIM={val_ssim:.4f} | "
            f"KL={avg_epoch_kl:.6f}"
        )

        with open(LOSS_LOG, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                f"{val_psnr:.6f},{val_ssim:.6f},{avg_epoch_kl:.6f}\n"
            )

        # ===== SAVE BEST =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_checkpoint(epoch, model, optimizer, best_val_loss, checkpoint_path)
            print("💾 Saved new best model!")
            counter = 0
        else:
            counter += 1
            if counter >= EARLY_STOPPING:
                print("⛔ Early stopping triggered!")
                break

        scheduler.step()

    print("✅ Training completed.")


if __name__ == "__main__":
    main()