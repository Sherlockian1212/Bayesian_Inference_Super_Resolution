from utils import config
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import data.ixi_mri_ful as ixi_mri_ful
import data.kirby_21 as kirby_21
import data.ood_data as ood

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_data_loader(data_loader):
    for batch_inputs, batch_targets in data_loader:
        print("✅ Sample batch from train_loader:")
        for i in range(min(3, len(batch_inputs))):
            print(f"Sample {i + 1}:")
            print(f"  Input shape:  {batch_inputs[i].shape}")
            print(f"  Target shape: {batch_targets[i].shape}")
            print(f"  Input tensor:  {batch_inputs[i][0, :, :].numpy()[:5, :5]}")
            print(f"  Target tensor: {batch_targets[i][0, :, :].numpy()[:5, :5]}")
            print(f"  Input min: {batch_inputs[i].min().item()}, max: {batch_inputs[i].max().item()}")
            print(f"  Target min: {batch_targets[i].min().item()}, max: {batch_targets[i].max().item()}")
        break

    for i in range(min(3, len(batch_inputs))):
        inp_tensor = batch_inputs[i].cpu()
        tgt_tensor = batch_targets[i].cpu()

        plt.figure(figsize=(6, 3))

        # Input image
        plt.subplot(1, 2, 1)
        if inp_tensor.shape[0] == 1:  # grayscale
            plt.imshow(inp_tensor[0], cmap='gray', interpolation='none')
        elif inp_tensor.shape[0] == 3:  # RGB
            plt.imshow(inp_tensor.permute(1, 2, 0), interpolation='none')
        plt.title("Input Image")
        plt.axis('off')

        # Target image
        plt.subplot(1, 2, 2)
        if tgt_tensor.shape[0] == 1:  # grayscale
            plt.imshow(tgt_tensor[0], cmap='gray', interpolation='none')
        elif tgt_tensor.shape[0] == 3:  # RGB
            plt.imshow(tgt_tensor.permute(1, 2, 0), interpolation='none')
        plt.title("Target Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def data_prepairing_pipeline(dataset, log_file=None, show=False, scale=config.DOWN_SCALE_RATE):
    if dataset == "ixi_full":
        print("Processing IXI dataset")
        train_dict, val_dict, test_dict = ixi_mri_ful.ixi_get_data_info_and_split(log_file=log_file)

        train_ds = ixi_mri_ful.IXI_MRI_Dataset_1(train_dict, scale=scale)
        val_ds = ixi_mri_ful.IXI_MRI_Dataset_1(val_dict, scale=scale)
        test_ds = ixi_mri_ful.IXI_MRI_Dataset_1(test_dict, scale=scale)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True)

    elif dataset == "kirby_21":
        print("Processing KIRBY_21 dataset")
        train_dict, val_dict, test_dict = kirby_21.kirby_get_data_info_and_split()

        train_ds = kirby_21.KIRBY_21_Dataset(train_dict, scale=scale)
        val_ds = kirby_21.KIRBY_21_Dataset(val_dict, scale=scale)
        test_ds = kirby_21.KIRBY_21_Dataset(test_dict, scale=scale)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True)

    elif dataset == "ood":
        print("Processing Out-of-Distribution Data")
        ood_data = ood.ood_get_data_info_and_split()
        ood_ds = ood.OOD_Dataset(ood_data, scale=scale)
        ood_loader = DataLoader(ood_ds, batch_size=16, num_workers=8, shuffle=False, pin_memory=True)
        return ood_loader

    if show:
        test_data_loader(test_loader)
    return train_loader, val_loader, test_loader

if __name__=="__main__":
    data_prepairing_pipeline("kirby_21", r"/logs/kirby_21_studies_log.txt", show=True)