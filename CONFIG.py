import os

IMG_SIZE = (512, 512)
BATCH_SIZE = 4

DATASET_PATH = os.path.join("dataset")
REAL_IMG_PATH = os.path.join(DATASET_PATH, "original")
MASK_IMG_PATH = os.path.join(DATASET_PATH, "mask")
DEVICE = "cuda"

IN_CH = 3
OUT_CH = 1
LEARNING_RATE = 1e-4
EPOCHS = 20

LOAD = True
MODEL_PATH = os.path.join("models", "resnet50_unet_10.pth")

MODEL_TYPE = "resnet50_unet" # "unet" or "resnet50_unet"