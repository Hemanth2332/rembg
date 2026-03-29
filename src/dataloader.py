import random
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

from src.CONFIG import MASK_IMG_PATH, REAL_IMG_PATH


class BgRemovalDataset(Dataset):
    def __init__(self, real_img_path, mask_img_path, crop_size=512, is_train=True):
        self.real_images = sorted(glob.glob(os.path.join(real_img_path, "*.jpg")))
        self.mask_images = sorted(glob.glob(os.path.join(mask_img_path, "*.png")))
        self.crop_size = crop_size
        self.is_train = is_train
        self.to_tensor = T.ToTensor()

        # Color augmentations (image only)
        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )

        assert len(self.real_images) == len(self.mask_images), print(len(self.real_images), len(self.mask_images))

    def __len__(self):
        return len(self.real_images)

    def random_crop(self, image, mask):
        w, h = image.size
        th, tw = self.crop_size, self.crop_size

        if w < tw or h < th:
            new_h = max(h, th)
            new_w = max(w, tw)

            image = F.resize(image, (new_h, new_w))
            mask = F.resize(mask, (new_h, new_w),
                            interpolation=F.InterpolationMode.NEAREST)

            w, h = image.size
        
        if not self.is_train:
            # For validation, use center crop or resize to crop_size directly
            # Here we follow the logic of resizing to crop_size if we're in validation
            image = F.resize(image, (th, tw))
            mask = F.resize(mask, (th, tw), interpolation=F.InterpolationMode.NEAREST)
            return image, mask

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        image = F.crop(image, i, j, th, tw)
        mask = F.crop(mask, i, j, th, tw)

        return image, mask

    def augment(self, image, mask):
        
        # Horizontal Flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        
        # Small Rotation
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        
        # Color Jitter (image only)
        if random.random() > 0.5:
            image = self.color_jitter(image)


        # Gaussian Blur (image only)
        if random.random() > 0.8:
            image = F.gaussian_blur(image, kernel_size=3)

        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.real_images[idx]).convert("RGB")
        mask = Image.open(self.mask_images[idx]).convert("L")

        # Random Crop (or Resize if validation)
        image, mask = self.random_crop(image, mask)

        # Augmentations (only during training)
        if self.is_train:
            image, mask = self.augment(image, mask)

        # To Tensor
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        # Binary mask
        mask = (mask > 0.5).float()

        return image, mask
    

if __name__ == "__main__":
    dataset = BgRemovalDataset(REAL_IMG_PATH, MASK_IMG_PATH)
    img, mask = dataset[0]

    print(img.shape, mask.shape)