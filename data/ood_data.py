from utils import config
import random
import os
from data.data_preprocessing import pre_procesing, downscale, get_slide_by_index_2
from torch.utils.data import Dataset
import nibabel as nib

def ood_get_data_info_and_split(folder_path=config.OOD_FOLDER_PATH):
    print("🔍 Scanning Out-of-Distribution dataset...\n")

    data_list = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            fpath = os.path.join(folder_path, fname)
            img = nib.load(fpath)
            data = img.get_fdata()

            # assume slicing along first axis
            num_slices = data.shape[0]

            for idx in range(num_slices):
            # for idx in range(min(10, num_slices), min(140, num_slices)):
                data_list.append({
                    "path": fpath,
                    "slice_idx": idx
                })

    print(f"✅ Found {len(data_list)} slices from OOD dataset")
    return data_list

def ood_pre_process(path, index, scale=config.DOWN_SCALE_RATE):
    target_tensor = get_slide_by_index_2(path, index)
    input_tensor = downscale(target_tensor, scale=scale)
    target_tensor = pre_procesing(target_tensor)
    input_tensor = pre_procesing(input_tensor)
    return input_tensor, target_tensor

class OOD_Dataset(Dataset):
    def __init__(self,
                 data_list,
                 scale=config.DOWN_SCALE_RATE):
        """
        data_list: list of dict {path, slice_idx}
        """
        self.data_list = data_list
        self.scale = scale

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        nii_file = sample["path"]
        slice_idx = sample["slice_idx"]

        x, y = ood_pre_process(nii_file, slice_idx, scale=self.scale)
        return x, y

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    print("Processing OOD dataset")
    ood_data = ood_get_data_info_and_split()

    ood_ds = OOD_Dataset(ood_data)

    ood_loader = DataLoader(ood_ds, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)