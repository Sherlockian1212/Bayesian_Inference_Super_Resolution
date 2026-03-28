from model.build_model import ModelSR
from utils.metrics import psnr, ssim
from data.data_preprocessing import pre_procesing, downscale
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from data.data_loader import data_prepairing_pipeline as dataset
from utils.helper import unwrap_generator_output
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

def evaluation(scale="x4", version="v1", data="kirby_21", n_samples=10, stat=False):
    """
    Evaluate the model by sampling multiple outputs (MC sampling)
    and computing mean ± std of PSNR and SSIM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    scale_factor = int(scale[1:])
    # Load data
    train_loader, val_loader, test_loader = dataset(
        dataset=data,
        log_file=rf"../logs/{data}_studies_log.txt",
        scale=scale_factor
    )

    # Load model
    model = ModelSR(scale=scale_factor).to(device)
    model.load_state_dict(torch.load(f"../weights/best_SR_{scale}_{version}_{data}.pth", map_location=device))
    model.eval()

    all_psnr = []
    all_ssim = []
    pearson_list, spearman_list = [], []
    patch8_p, patch8_s = [], []
    patch16_p, patch16_s = [], []
    patch32_p, patch32_s = [], []

    with torch.no_grad():
        for lr_img, hr_img in tqdm(test_loader, desc="Evaluating", leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            preds = []
            # Multiple forward passes for uncertainty estimation
            for _ in range(n_samples):
                g_out = model(lr_img)
                sr_img, mu, sigma = unwrap_generator_output(g_out)
                preds.append(sr_img)

            # Stack predictions: [n_samples, B, C, H, W]
            preds = torch.stack(preds, dim=0)

            # Mean prediction
            sr_mean = preds.mean(dim=0)

            # Uncertainty map
            sr_std = preds.std(dim=0)

            # Compute per-batch metrics
            batch_psnr = psnr(sr_mean, hr_img)
            batch_ssim = ssim(sr_mean, hr_img).item()

            all_psnr.append(batch_psnr)
            all_ssim.append(batch_ssim)

            # Compute uncertainty correlations
            if stat:
                B = hr_img.shape[0]
                for b in range(B):
                    hr_np = hr_img[b].detach().cpu().numpy()
                    sr_np = sr_mean[b].detach().cpu().numpy()
                    std_np = sr_std[b].detach().cpu().numpy()

                    pearson_corr_log, pearson_corr,spearman_corr_log, spearman_corr, pearson_patch_8, spearman_patch_8, pearson_patch_16, spearman_patch_16, pearson_patch_32, spearman_patch_32 = uncertainty(
                        hr_np, sr_np, std_np, plot=False
                    )

                    pearson_list.append(pearson_corr)
                    spearman_list.append(spearman_corr)
                    patch8_p.append(pearson_patch_8)
                    patch8_s.append(spearman_patch_8)
                    patch16_p.append(pearson_patch_16)
                    patch16_s.append(spearman_patch_16)
                    patch32_p.append(pearson_patch_32)
                    patch32_s.append(spearman_patch_32)

    all_psnr, all_ssim = np.array(all_psnr), np.array(all_ssim)

    def mean_std(x): return np.mean(x), np.std(x)

    psnr_m, psnr_s = mean_std(all_psnr)
    ssim_m, ssim_s = mean_std(all_ssim)

    print("\n🧪 Test Results Summary:")
    print(f"PSNR = {psnr_m:.4f} ± {psnr_s:.4f}")
    print(f"SSIM = {ssim_m:.4f} ± {ssim_s:.4f}")

    if stat:
        p_m, p_s = mean_std(pearson_list)
        s_m, s_s = mean_std(spearman_list)
        p8_m, p8_s = mean_std(patch8_p)
        s8_m, s8_s = mean_std(patch8_s)
        p16_m, p16_s = mean_std(patch16_p)
        s16_m, s16_s = mean_std(patch16_s)
        p32_m, p32_s = mean_std(patch32_p)
        s32_m, s32_s = mean_std(patch32_s)
        print(f"Uncertainty Pearson = {p_m:.4f} ± {p_s:.4f}")
        print(f"Uncertainty Spearman = {s_m:.4f} ± {s_s:.4f}")
        print(f"Patchwise (8x8) Pearson = {p8_m:.4f} ± {p8_s:.4f}")
        print(f"Patchwise (8x8) Spearman = {s8_m:.4f} ± {s8_s:.4f}")
        print(f"Patchwise (16x16) Pearson = {p16_m:.4f} ± {p16_s:.4f}")
        print(f"Patchwise (16x16) Spearman = {s16_m:.4f} ± {s16_s:.4f}")
        print(f"Patchwise (32x32) Pearson = {p32_m:.4f} ± {p32_s:.4f}")
        print(f"Patchwise (32x32) Spearman = {s32_m:.4f} ± {s32_s:.4f}")

def visualize(scale="x4", version="v1", data="ixi_full"):
    path=f"../logs/loss_log_{scale}_{version}_{data}.txt"
    df = pd.read_csv(path)
    epochs = df['Epoch']

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # --- Train & Val Loss ---
    axs[0, 0].plot(epochs, df['Train_G_Loss'], label='Train G Loss')
    axs[0, 0].plot(epochs, df['Train_D_Loss'], label='Train D Loss')
    axs[0, 0].plot(epochs, df['Val_Loss'], label='Val Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Losses')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- KL ---
    axs[0, 1].plot(epochs, df['Avg_KL'], color='red')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('KL Loss')
    axs[0, 1].set_title('KL Loss per Epoch')
    axs[0, 1].grid(True)

    # --- PSNR ---
    axs[1, 1].plot(epochs, df['Val_PSNR'], color='green')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('PSNR')
    axs[1, 1].set_title('PSNR per Epoch')
    axs[1, 1].grid(True)

    # --- SSIM ---
    axs[1, 0].plot(epochs, df['Val_SSIM'], color='orange')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('SSIM')
    axs[1, 0].set_title('SSIM per Epoch')
    axs[1, 0].grid(True)

    # --- Title ---
    fig.suptitle(f'Training Progress of Bayesian_GAN ({scale})', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def patchwise_correlation(u, e, patch_size=32, plot=True):
    H, W = u.shape
    ph, pw = patch_size, patch_size

    # Cắt patch thành từng vùng
    patches_u, patches_e = [], []
    for i in range(0, H, ph):
        for j in range(0, W, pw):
            u_patch = u[i:i + ph, j:j + pw]
            e_patch = e[i:i + ph, j:j + pw]
            if u_patch.size == 0 or e_patch.size == 0:
                continue
            patches_u.append(np.mean(u_patch))
            patches_e.append(np.mean(e_patch))

    patches_u = np.array(patches_u)
    patches_e = np.array(patches_e)

    # Tính tương quan giữa các patch
    pearson_corr, _ = pearsonr(patches_u, patches_e)
    spearman_corr, _ = spearmanr(patches_u, patches_e)

    if plot:
        print(f"Patchwise Pearson Corr (Patch {patch_size}): {pearson_corr:.4f}")
        print(f"Patchwise Spearman Corr (Patch {patch_size}): {spearman_corr:.4f}")

    return pearson_corr, spearman_corr

def uncertainty(hr_image, sr_image, uncertainty_map, plot=True):
    # --- Convert to numpy ---
    hr = hr_image if isinstance(hr_image, np.ndarray) else hr_image.detach().cpu().numpy()
    sr = sr_image if isinstance(sr_image, np.ndarray) else sr_image.detach().cpu().numpy()
    std = uncertainty_map if isinstance(uncertainty_map, np.ndarray) else uncertainty_map.detach().cpu().numpy()

    # --- Ensure 2D shape (H, W) ---
    hr = np.squeeze(hr)
    sr = np.squeeze(sr)
    std = np.squeeze(std)

    # --- Compute per-pixel absolute error ---
    error = np.abs(hr - sr)

    # --- Flatten for correlation ---
    u = std.flatten()
    e = error.flatten()

    # --- Correlation metrics ---
    pearson_corr_log, _ = pearsonr(np.log(u+1e-8), e)
    pearson_corr, _ = pearsonr(u, e)
    spearman_corr_log, _ = spearmanr(np.log(u+1e-8), e)
    spearman_corr, _ = spearmanr(u, e)

    pearson_patch_8, spearman_patch_8 = patchwise_correlation(std, error, patch_size=8, plot=plot)
    pearson_patch_16, spearman_patch_16 = patchwise_correlation(std, error, patch_size=16, plot=plot)
    pearson_patch_32, spearman_patch_32 = patchwise_correlation(std, error, patch_size=32, plot=plot)

    # --- Optional scatter plot ---
    if plot:
        print(f"Pearson Corr Log (Uncertainty ↔ Error): {pearson_corr_log:.4f}")
        print(f"Pearson Corr (Uncertainty ↔ Error): {pearson_corr:.4f}")
        print(f"Spearman Corr Log (Rank Consistency): {spearman_corr_log:.4f}")
        print(f"Spearman Corr (Rank Consistency): {spearman_corr:.4f}")
        plt.figure(figsize=(6, 5))
        plt.scatter(u, e, s=3, alpha=0.3)
        plt.xlabel("Uncertainty (std_map)")
        plt.ylabel("Pixel Error (|HR - SR|)")
        plt.title(f"Pearson={pearson_corr_log:.3f}, Spearman={spearman_corr:.3f}")
        plt.tight_layout()
        plt.show()

    return pearson_corr_log, pearson_corr,spearman_corr_log, spearman_corr, pearson_patch_8, spearman_patch_8, pearson_patch_16, spearman_patch_16, pearson_patch_32, spearman_patch_32

def predict(scale="x4", version="v1", model_data = "ixi_full",
            # path=r"D:\Datasets\Kirby_21\KKI2009-07\T1\KKI2009-07-MPRAGE.nii",  data_name="kirby_21",
            # path=r"D:\Datasets\IXI_MRI_FULL\126_HH\T1\IXI126-HH-1437-T1.nii",  data_name="ixi_full",
            path=r"D:\Datasets\Openneuro\sub-BA4_run-02_T1w.nii", data_name="other",
            slice_index=100, n_samples=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_factor = int(scale[1:])

    model = ModelSR(scale=scale_factor).to(device)
    model.load_state_dict(torch.load(f"../weights/best_SR_{scale}_{version}_{model_data}.pth", map_location=device))
    model.eval()

    image = nib.load(path)
    data = image.get_fdata()
    if data_name == "ixi_full":
        img_slice = data[:, :, slice_index]
    elif data_name == "kirby_21":
        img_slice = data[slice_index, :, :]
    else: # other dataset
        img_slice = data[slice_index, :, :]
        img_slice = np.rot90(img_slice, k=2).copy()
        img_slice = np.fliplr(img_slice).copy()

    image_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    downscale_tensor = downscale(image_tensor, scale=scale_factor)
    downscale_tensor = downscale_tensor.unsqueeze(0)
    image_norm = pre_procesing(downscale_tensor).to(device)  # normalized 0-1
    target_tensor = pre_procesing(image_tensor.clone())

    preds = []
    aleatorics = []
    with torch.no_grad():
        for _ in range(n_samples):
            g_out = model(image_norm.to(device))
            out, mu, sigma = unwrap_generator_output(g_out)
            preds.append(out.squeeze(0).cpu().numpy())
            aleatorics.append(sigma.squeeze(0).cpu().numpy())

    preds = np.stack(preds, axis=0)
    aleatorics = np.stack(aleatorics, axis=0)

    # Calculation output and uncertainty
    output_np = preds.mean(axis=0).squeeze(0)
    epistemic_map = preds.std(axis=0).squeeze(0)
    aleatoric_map = aleatorics.mean(axis=0).squeeze(0)

    # Calculation Error
    input_np = image_norm.squeeze(0).squeeze(0).cpu().numpy()
    target_np = target_tensor.squeeze(0).squeeze(0).cpu().numpy()
    err = np.abs(output_np - target_np)

    # Calculation Interpolation
    up = cv2.resize(input_np, (input_np.shape[1] * scale_factor, input_np.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
    psnr_val = peak_signal_noise_ratio(target_np, output_np, data_range=1.0)
    ssim_val = structural_similarity(target_np, output_np, data_range=1.0)
    psnr_val_INTER_CUBIC = peak_signal_noise_ratio(target_np, up, data_range=1.0)
    ssim_val_INTER_CUBIC = structural_similarity(target_np, up, data_range=1.0)
    print(f"📊 PSNR: {psnr_val:.2f} dB")
    print(f"📊 SSIM: {ssim_val:.4f}")
    print(f"📊 PSNR-INTER_CUBIC: {psnr_val_INTER_CUBIC:.2f} dB")
    print(f"📊 SSIM-INTER_CUBIC: {ssim_val_INTER_CUBIC:.4f}")

    # Statistical Uncertainty
    alea = torch.from_numpy(aleatoric_map).to(device).float()
    alea = torch.log1p(alea)
    if isinstance(epistemic_map, np.ndarray):
        epis = torch.from_numpy(epistemic_map)
    device = epistemic_map.device
    alea = alea.to(device)
    epis = epis.to(device)
    sigma_total = torch.sqrt(epis ** 2 + alea ** 2)
    uncertainty(target_np, output_np, sigma_total)

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].imshow(input_np, cmap='gray')
    axs[0, 0].set_title("Input")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(output_np, cmap='gray')
    axs[0, 1].set_title(f"Predicted Mean\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(target_np, cmap='gray')
    axs[0, 2].set_title("Target")
    axs[0, 2].axis('off')

    im1 = axs[1, 0].imshow(epistemic_map, cmap='hot')
    axs[1, 0].set_title("Epistomic Uncertainty Map")
    axs[1, 0].axis('off')
    fig.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im2 = axs[1, 1].imshow(aleatoric_map, cmap='hot')
    axs[1, 1].set_title("Aleatoric Uncertainty Map")
    axs[1, 1].axis('off')
    fig.colorbar(im2, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im3 = axs[1, 2].imshow(err, cmap='hot')
    axs[1, 2].set_title("Error Map")
    axs[1, 2].axis('off')
    fig.colorbar(im3, ax=axs[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return output_np, epistemic_map, mu, sigma

def evaluation_ood(scale="x4", version="v1", model_data="kirby_21", n_samples=10, stat=False):
    """
        Evaluate the model by sampling multiple outputs (MC sampling)
        and computing mean ± std of PSNR and SSIM.
        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_factor = int(scale[1:])
    # Load data
    test_loader = dataset(
        dataset="ood",
        scale=scale_factor
    )
    # Load model
    model = ModelSR(scale=scale_factor).to(device)
    model.load_state_dict(torch.load(f"../weights/best_SR_{scale}_{version}_{model_data}.pth", map_location=device))
    model.eval()

    all_psnr = []
    all_ssim = []
    pearson_list, spearman_list = [], []
    patch8_p, patch8_s = [], []
    patch16_p, patch16_s = [], []
    patch32_p, patch32_s = [], []

    with torch.no_grad():
        for lr_img, hr_img in tqdm(test_loader, desc="Evaluating", leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            preds = []
            sigma_maps = []
            # Multiple forward passes for uncertainty estimation
            for _ in range(n_samples):
                g_out = model(lr_img)
                sr_img, mu, sigma = unwrap_generator_output(g_out)
                preds.append(sr_img)
                sigma_maps.append(sigma)

            # Stack predictions: [n_samples, B, C, H, W]
            preds = torch.stack(preds, dim=0)
            sigma_maps = torch.stack(sigma_maps, dim=0)

            # Mean prediction
            sr_mean = preds.mean(dim=0)

            # Uncertainty map
            sr_std = preds.std(dim=0)
            sigma_mean = sigma_maps.mean(dim=0)

            # Compute per-batch metrics
            batch_psnr = psnr(sr_mean, hr_img)
            batch_ssim = ssim(sr_mean, hr_img).item()

            all_psnr.append(batch_psnr)
            all_ssim.append(batch_ssim)

            # Compute uncertainty correlations
            if stat:
                B = hr_img.shape[0]
                for b in range(B):
                    hr_np = hr_img[b].detach().cpu().numpy()
                    sr_np = sr_mean[b].detach().cpu().numpy()
                    std_np = sr_std[b].detach().cpu().numpy()
                    sigma_np = sigma_mean[b].detach().cpu().numpy()

                    pearson_corr_log, pearson_corr, spearman_corr_log, spearman_corr, pearson_patch_8, spearman_patch_8, pearson_patch_16, spearman_patch_16, pearson_patch_32, spearman_patch_32 = uncertainty(
                        hr_np, sr_np, sigma_np, plot=False
                    )

                    pearson_list.append(pearson_corr)
                    spearman_list.append(spearman_corr)
                    patch8_p.append(pearson_patch_8)
                    patch8_s.append(spearman_patch_8)
                    patch16_p.append(pearson_patch_16)
                    patch16_s.append(spearman_patch_16)
                    patch32_p.append(pearson_patch_32)
                    patch32_s.append(spearman_patch_32)

    all_psnr, all_ssim = np.array(all_psnr), np.array(all_ssim)

    def mean_std(x):
        return np.mean(x), np.std(x)

    psnr_m, psnr_s = mean_std(all_psnr)
    ssim_m, ssim_s = mean_std(all_ssim)

    print("\n🧪 Test Results Summary:")
    print(f"PSNR = {psnr_m:.4f} ± {psnr_s:.4f}")
    print(f"SSIM = {ssim_m:.4f} ± {ssim_s:.4f}")

    if stat:
        p_m, p_s = mean_std(pearson_list)
        s_m, s_s = mean_std(spearman_list)
        p8_m, p8_s = mean_std(patch8_p)
        s8_m, s8_s = mean_std(patch8_s)
        p16_m, p16_s = mean_std(patch16_p)
        s16_m, s16_s = mean_std(patch16_s)
        p32_m, p32_s = mean_std(patch32_p)
        s32_m, s32_s = mean_std(patch32_s)
        print(f"Uncertainty Pearson = {p_m:.4f} ± {p_s:.4f}")
        print(f"Uncertainty Spearman = {s_m:.4f} ± {s_s:.4f}")
        print(f"Patchwise (8x8) Pearson = {p8_m:.4f} ± {p8_s:.4f}")
        print(f"Patchwise (8x8) Spearman = {s8_m:.4f} ± {s8_s:.4f}")
        print(f"Patchwise (16x16) Pearson = {p16_m:.4f} ± {p16_s:.4f}")
        print(f"Patchwise (16x16) Spearman = {s16_m:.4f} ± {s16_s:.4f}")
        print(f"Patchwise (32x32) Pearson = {p32_m:.4f} ± {p32_s:.4f}")
        print(f"Patchwise (32x32) Spearman = {s32_m:.4f} ± {s32_s:.4f}")

if __name__ == "__main__":
    visualize(scale="x4", version="v1", data="kirby_21")
    # predict(scale="x2", version="v1", model_data = "ixi_full")
    # predict(scale="x2", version="v1", model_data = "kirby_21")

    # evaluation(scale="x2", version="v1", data="ixi_full", n_samples=5, stat=True)
    # evaluation(scale="x4", version="v1", data="ixi_full", n_samples=5, stat=True)

    # evaluation_ood(scale="x2", version="v1", model_data="kirby_21", n_samples=5, stat=True)
    # evaluation_ood(scale="x2", version="v1", model_data="kirby_21", n_samples=10, stat=False)
    # evaluation_ood(scale="x2", version="v1", model_data="kirby_21", n_samples=20, stat=False)
    # evaluation_ood(scale="x4", version="v1", model_data="kirby_21", n_samples=5, stat=False)
    # evaluation_ood(scale="x4", version="v1", model_data="kirby_21", n_samples=10, stat=False)
    # evaluation_ood(scale="x4", version="v1", model_data="kirby_21", n_samples=20, stat=False)

    # evaluation_ood(scale="x2", version="v1", model_data="ixi_full", n_samples=5, stat=True)
    # evaluation_ood(scale="x2", version="v1", model_data="ixi_full", n_samples=10, stat=False)
    # evaluation_ood(scale="x2", version="v1", model_data="ixi_full", n_samples=20, stat=False)
    # evaluation_ood(scale="x4", version="v1", model_data="ixi_full", n_samples=5, stat=False)
    # evaluation_ood(scale="x4", version="v1", model_data="ixi_full", n_samples=10, stat=False)
    # evaluation_ood(scale="x4", version="v1", model_data="ixi_full", n_samples=20, stat=False)



