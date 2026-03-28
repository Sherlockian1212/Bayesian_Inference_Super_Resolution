IXI_FOLDER_PATH_FULL = r"D:\Datasets\IXI_MRI_FULL"
IXI_TYPE_FUll = ["T1", "T2"]

KIRBY_21_FOLDER_PATH_FULL = r"D:\Datasets\Kirby_21"
KIRBY_21_TYPE_FUll = ["T1", "T2"]
OOD_FOLDER_PATH = r"D:\Datasets\Openneuro"

NORMALIZATION = False

CHANNELS = 1

DOWN_SCALE_RATE = 2
LEARNING_RATE = 1e-4  # 2e-4

BATCH_SIZE = 16
EPOCH = 1000
EARLY_STOPPING = 10
PIXEL_VALUE_MAX = 1.0
BETA_KL = 1e-6
BETA_NLL = 1e-6

DROP_OUT = 0.2
PRETRAIN_EPOCH = 5
VERSION = "v1"
# DATA = "kirby_21"
DATA = "ixi_full"
DATA_LOG = rf"D:\PythonProject\SR\logs\{DATA}_studies_log.txt"

IXI_STUDY_LOG = rf"D:\PythonProject\SR\logs\ixi_full_studies_log.txt"
KIRBY_STUDY_LOG = rf"D:\PythonProject\SR\logs\kirby_21_studies_log.txt"

LOSS_LOG = f"../logs/loss_log_x{DOWN_SCALE_RATE}_{VERSION}_{DATA}.txt"
