import os

IMG_SIZE = (512, 512)
BATCH_SIZE = 4

DATASET_PATH = os.path.join(r"D:\coding\python_projects\pytorch\rembg\dut-dataset\DUTS-TR")
REAL_IMG_PATH = os.path.join(DATASET_PATH, "DUTS-TR-Image")
MASK_IMG_PATH = os.path.join(DATASET_PATH, "DUTS-TR-Mask")
DEVICE = "cuda"

IN_CH = 3
OUT_CH = 1
LEARNING_RATE = 1e-4
EPOCHS = 20

LOAD = False
MODEL_PATH = os.path.join("dut-models", "resnet50_unet_dut_10.pth")

TEST_DATASET_PATH = os.path.join(r"D:\coding\python_projects\pytorch\rembg\dut-dataset\DUTS-TE")
TEST_REAL_IMG_PATH = os.path.join(TEST_DATASET_PATH, "DUTS-TE-Image")
TEST_MASK_IMG_PATH = os.path.join(TEST_DATASET_PATH, "DUTS-TE-Mask")

LOG_DIR = "logs"

MODEL_TYPE = "resnet50_unet" # "unet" or "resnet50_unet"