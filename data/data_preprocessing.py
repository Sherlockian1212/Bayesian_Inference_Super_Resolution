import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import gzip
import torch
import torch.nn.functional as F
import numpy as np
from utils.config import NORMALIZATION

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def mri_info(path=r"D:\Datasets\IXI_MRI_FULL\126_HH\T1\IXI126-HH-1437-T1.nii"):
# def mri_info(path=r"D:\Datasets\Kirby_21\KKI2009-01\T1\KKI2009-01-MPRAGE.nii"):
def mri_info(path=r"D:\Datasets\Openneuro\sub-BA2_run-02_T1w.nii"):
    img = nib.load(path)
    data = img.get_fdata()

    print("Filename:", img.get_filename())
    print("Data type:", img.get_data_dtype())
    print("Shape (H x W x Depth):", img.shape)  # kích thước ảnh (số lát cắt = depth)
    print("Voxel resolution (mm):", img.header.get_zooms())
    print("Affine matrix:\n", img.affine)
    print("\nVoxel intensity stats:")
    print("Min:", data.min())
    print("Max:", data.max())
    print("Mean:", data.mean())
    print("Std Dev:", data.std())
    print("\nHeader:")
    print(img.header)

    slice_index = 50
    image_tensor = torch.tensor(data[:, :, slice_index], dtype=torch.float32)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(_normalize_single(image_tensor), cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.hist(data.ravel(), bins=100, color='blue', alpha=0.7)
    plt.title("Voxel Intensity Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print(data.shape)
    for slice_index in range(data.shape[0]):  # trục z
        # image_tensor = torch.tensor(data[:, :, slice_index], dtype=torch.float32)
        image_tensor = torch.tensor(data[slice_index, :, :], dtype=torch.float32)
        plt.imshow(_normalize_single(image_tensor), cmap='gray')
        plt.title(f"Slice {slice_index + 1}/{data.shape[2]}")
        plt.axis('off')
        plt.pause(0.1)
        plt.clf()
    plt.close()

def read_ppm(file_path=r"D:\Datasets\STARE\im0091.ppm"):
    # Read PPM image using Pillow
    img = Image.open(file_path)

    # Get image info
    width, height = img.size
    mode = img.mode  # RGB, L, etc.
    file_size_kb = os.path.getsize(file_path) / 1024

    print(f"📏 Resolution: {width}x{height} pixels")
    print(f"🎨 Channels: {mode} ({'Color' if mode == 'RGB' else 'Grayscale'})")
    print(f"💾 File size: {file_size_kb:.2f} KB")

    # Display image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    return img

def organize_nifti_files(source_dir, dest_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                try:
                    # Tách các thành phần từ đường dẫn
                    parts = root.split(os.sep)
                    study = parts[-3]  # Lấy {Study}
                    folder_name = os.path.splitext(os.path.splitext(file)[0])[0]

                    # Tạo đường dẫn đích
                    dest_study_dir = os.path.join(dest_dir, "T1", study)
                    os.makedirs(dest_study_dir, exist_ok=True)

                    # Tạo đường dẫn tệp nguồn và đích
                    source_file_path = os.path.join(root, file)
                    file_name = os.path.splitext(folder_name)[0]
                    dest_file_path = os.path.join(dest_study_dir, f"{file_name}.nii")

                    # Giải nén tệp .gz
                    with gzip.open(source_file_path, 'rb') as f_in:
                        with open(dest_file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    print(f"Extracted and copy: {source_file_path} -> {dest_file_path}")

                except (IndexError, nib.filebasedimages.ImageFileError) as e:
                    print(f"Error to solve at {file}: {e}")

def get_slices(path, start=50, end=100):
    img = nib.load(path)
    data = img.get_fdata()

    slices = data[:, :, start:end]

    # Convert to PyTorch tensor and rearrange to (N, 1, H, W)
    tensor = torch.from_numpy(slices).float()         # shape: (H, W, N)
    tensor = tensor.permute(2, 0, 1).unsqueeze(1)      # shape: (N, 1, H, W)

    return tensor

def get_slide_by_index(path, index):
    img = nib.load(path)
    data = img.get_fdata()
    slice_2d = data[:, :, index]  # shape: (H, W)
    tensor = torch.from_numpy(slice_2d).unsqueeze(0)          # shape: (1, 1, H, W)
    return tensor

def get_slide_by_index_1(path, index):
    img = nib.load(path)
    data = img.get_fdata()
    slice_2d = data[index, :, :]  # shape: (H, W)
    tensor = torch.from_numpy(slice_2d).unsqueeze(0)          # shape: (1, 1, H, W)
    return tensor

def get_slide_by_index_2(path, index):
    img = nib.load(path)
    data = img.get_fdata()
    slice_2d = data[index, :, :]  # shape: (H, W)
    slice_2d = np.rot90(slice_2d, k=2).copy()
    slice_2d = np.fliplr(slice_2d).copy()
    tensor = torch.from_numpy(slice_2d).unsqueeze(0)  # shape: (1, 1, H, W)
    return tensor


# def _normalize_single(img: torch.Tensor, clip_percentile=(1, 99)) -> torch.Tensor:
#     """Normalize a single MRI volume [C,H,W]"""
#     flat = img.contiguous().view(-1)
#     low = torch.quantile(flat, clip_percentile[0] / 100.0)
#     high = torch.quantile(flat, clip_percentile[1] / 100.0)
#
#     img = torch.clamp(img, min=low.item(), max=high.item())
#
#     min_val = img.min()
#     max_val = img.max()
#     if max_val > min_val:
#         img = (img - min_val) / (max_val - min_val)
#         img = img * PIXEL_VALUE_MAX
#     else:
#         img = torch.zeros_like(img)
#
#     return img

def _normalize_single(img: torch.Tensor, clip_percentile=(1, 99)) -> torch.Tensor:
    """Normalize a single MRI volume [C,H,W]"""
    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)

    img = torch.clamp(img, min=0.0, max=1.0)

    return img

def gaussian_kernel(kernel_size=3, sigma=0.1, channels=1):
    # Tạo Gaussian kernel 2D
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)  # cho multi-channel
    return kernel

def denoise_gaussian(image: torch.Tensor, kernel_size=5, sigma=1):
    """
    Lọc nhiễu Gaussian blur bằng conv2d
    image: [C,H,W] hoặc [B,C,H,W]
    """
    if image.dim() == 3:  # [C,H,W] -> [1,C,H,W]
        image = image.unsqueeze(0)

    b, c, h, w = image.shape
    kernel = gaussian_kernel(kernel_size, sigma, c).to(image.device)
    padding = kernel_size // 2
    out = F.conv2d(image, kernel, padding=padding, groups=c)
    return out.squeeze(0) if b == 1 else out

def pre_procesing(image: torch.Tensor, clip_percentile=(1, 99), denoise=NORMALIZATION) -> torch.Tensor:
    """
    Pre-process MRI images for Super-Resolution:
    - Clip outliers by percentile
    - Normalize to [0, 1] per image
    Args:
        image (torch.Tensor): Input tensor of shape [C, H, W] or [B, C, H, W]
        clip_percentile (tuple): Percentiles for clipping (low, high)
    Returns:
        torch.Tensor: Normalized tensor [0, 1] with same shape as input
    """
    image = image.float()

    if image.dim() == 3:  # [C, H, W]
        image = _normalize_single(image, clip_percentile)
        if denoise:
            image = denoise_gaussian(image)
    elif image.dim() == 4:  # [B, C, H, W]
        processed = []
        for img in image:  # img shape: [C, H, W]
            img = _normalize_single(img, clip_percentile)
            if denoise:
                img = denoise_gaussian(img)
            processed.append(img)
        image = torch.stack(processed, dim=0)
    else:
        raise ValueError("Input tensor must be [C,H,W] or [B,C,H,W]")
    return image

def downscale(image: torch.Tensor, scale: int = 2, mode: str = 'bilinear') -> torch.Tensor:
    """
    Downscale an image tensor by a given scale factor.

    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W] or [B, C, H, W].
        scale (int): Scale factor to downscale.
        mode (str): Interpolation mode (linear | bilinear | bicubic | trilinear, etc.).

    Returns:
        torch.Tensor: Downscaled image tensor (same shape type as input).
    """
    # Add batch dimension if missing
    if image.dim() == 3:
        image = image.unsqueeze(0)

    _, _, h, w = image.shape
    new_h, new_w = h // scale, w // scale

    # Resize using interpolation
    downscaled = F.interpolate(image, size=(new_h, new_w), mode=mode, align_corners=False)

    # Remove batch dimension if it was originally 3D
    return downscaled.squeeze(0) if downscaled.shape[0] == 1 else downscaled

def upscale(image: torch.Tensor, scale: int = 2, mode: str = 'bicubic') -> torch.Tensor:
    """
    Upscale an image tensor by a given scale factor.

    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W] or [B, C, H, W].
        scale (int): Scale factor to upscale.
        mode (str): Interpolation mode ('nearest', 'bilinear', 'bicubic', etc.).

    Returns:
        torch.Tensor: Upscaled image tensor (same shape type as input).
    """
    # Add batch dimension if missing
    if image.dim() == 3:
        image = image.unsqueeze(0)

    _, _, h, w = image.shape
    new_h, new_w = h * scale, w * scale

    # Resize using interpolation
    upscaled = F.interpolate(image, size=(new_h, new_w), mode=mode, align_corners=False)

    # Remove batch dimension if it was originally 3D
    return upscaled.squeeze(0) if upscaled.shape[0] == 1 else upscaled

if __name__=="__main__":
    mri_info()
    # organize_nifti_files(r"D:\Datasets\guest-20250802_100654_ex", r"D:\Datasets\IXI_MRI")