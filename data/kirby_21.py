from utils import config
import random
import os
from data.data_preprocessing import pre_procesing, downscale, get_slide_by_index_1
from torch.utils.data import Dataset
import nibabel as nib

def kirby_get_data_info_and_split(folder_path=config.KIRBY_21_FOLDER_PATH_FULL,
                                log_file = None,
                                train_ratio=0.6,
                                val_ratio=0.2,
                                test_ratio=0.2,
                                seed=42):
    if log_file == None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        random.seed(seed)

        print("🔍 Scanning KIRBY_21 dataset...\n")

        studies = [study for study in os.listdir(folder_path)]

        total = len(studies)

        studies = sorted(studies)
        random.shuffle(studies)

        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        train_study = studies[:n_train]
        val_study = studies[n_train:n_train + n_val]
        test_study = studies[n_train + n_val:]

        print(f"✅ Total={total}, train={len(train_study)}, val={len(val_study)}, test={len(test_study)}")

        log_path = config.KIRBY_STUDY_LOG
        with open(log_path, "w") as f:
            f.write("=== TRAIN IMAGES ===\n")
            f.write(", ".join(train_study) + "\n")

            f.write("\n=== VAL IMAGES ===\n")
            f.write(", ".join(val_study) + "\n")

            f.write("\n=== TEST IMAGES ===\n")
            f.write(", ".join(test_study) + "\n")

        print(f"\n📝 Study splits logged to: {log_path}")
    else:
        # --- Load from existing log file ---
        log_path = log_file
        assert os.path.exists(log_path), f"Log file not found: {log_path}"

        train_study, val_study, test_study = [], [], []
        current = None
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("=== TRAIN"):
                    current = "train"
                elif line.startswith("=== VAL"):
                    current = "val"
                elif line.startswith("=== TEST"):
                    current = "test"
                elif line != "":
                    studies = [s.strip() for s in line.split(",") if s.strip()]
                    if current == "train":
                        train_study.extend(studies)
                    elif current == "val":
                        val_study.extend(studies)
                    elif current == "test":
                        test_study.extend(studies)

        print(f"📂 Loaded study splits from {log_path}")
        print(f"✅ train={len(train_study)}, val={len(val_study)}, test={len(test_study)}")

    return train_study, val_study, test_study

def kirby_pre_process(path, index, scale=config.DOWN_SCALE_RATE):
    target_tensor = get_slide_by_index_1(path, index)
    input_tensor = downscale(target_tensor, scale=scale)
    target_tensor = pre_procesing(target_tensor)
    input_tensor = pre_procesing(input_tensor)
    return input_tensor, target_tensor

class KIRBY_21_Dataset(Dataset):
    def __init__(self, data_dict,
                 path=config.KIRBY_21_FOLDER_PATH_FULL,
                 scan_types = config.KIRBY_21_TYPE_FUll,
                 scale = config.DOWN_SCALE_RATE):
        self.data_dict = data_dict
        self.folder_path = path
        self.scan_types = scan_types
        self.scale = scale
        self.img_list = self.get_list() # [<StudyID>_<ScanType>_<Slice>]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]  # [<StudyID>_<ScanType>_<Slice>]

        study, scan_type, slice_idx = img_id.rsplit("_", 2)
        slice_idx = int(slice_idx)

        scan_dir = os.path.join(self.folder_path, study, scan_type)
        nii_files = [f for f in os.listdir(scan_dir) if f.endswith(".nii")]
        if not nii_files:
            raise FileNotFoundError(f"No .nii file in {scan_dir}")

        nii_file = os.path.join(scan_dir, nii_files[0])
        x, y = kirby_pre_process(nii_file, slice_idx, scale=self.scale)
        return x, y

    def get_list(self):
        studies = []
        for study in self.data_dict:
            for scan_type in self.scan_types:
                scan_dir = os.path.join(self.folder_path, study, scan_type)
                if not os.path.isdir(scan_dir):
                    continue

                nii_files = [f for f in os.listdir(scan_dir) if f.endswith(".nii")]
                if not nii_files:
                    continue

                nii_file = os.path.join(scan_dir, nii_files[0])
                try:
                    img = nib.load(nii_file)
                    n_slices = img.shape[0]  # get number of slices
                except Exception as e:
                    print(f"Error reading {nii_file}: {e}")
                    continue

                for i in range(min(10, n_slices), min(140, n_slices)):
                # for i in range(n_slices):
                    studies.append(f"{study}_{scan_type}_{i}")

        print(f"Total samples: {len(studies)}")
        return studies

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    print("Processing KIRBY_21 dataset")
    train_dict, val_dict, test_dict = kirby_get_data_info_and_split()

    train_ds = KIRBY_21_Dataset(train_dict)
    val_ds = KIRBY_21_Dataset(val_dict)
    test_ds = KIRBY_21_Dataset(test_dict)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, pin_memory=True)
