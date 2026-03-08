import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from src.CONFIG import *
from src.model import ResNet50_UNet

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path='models/resnet50_unet_20.pth'):
    """Loads the ResNet50_UNet model with pre-trained weights."""
    model = ResNet50_UNet(in_ch=IN_CH, out_ch=OUT_CH, pretrained=False).to(DEVICE)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    model.eval()
    return model

def remove_artifacts(mask):
    """Keeps only largest connected component from a binary mask."""
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = (labels == largest_label).astype(np.uint8) * 255
    return cleaned_mask

def process_image(model, input_img):
    """Processes an input image array and returns the predicted mask and foreground image."""
    if input_img is None:
        return None, None
    
    # Original image for reference
    original_pil = Image.fromarray(input_img)
    size = original_pil.size
    
    # Preprocess
    img = original_pil.resize((512, 512))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        pred_mask = model(img_tensor)
        pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
    
    # Postprocess
    pred_mask = (pred_mask > 0.5).astype('uint8') * 255
    pred_mask = remove_artifacts(pred_mask)
    mask_pil = Image.fromarray(pred_mask).resize(size)
    
    # Generate Transparent Result (Foreground)
    foreground = original_pil.copy()
    foreground.putalpha(mask_pil)
    
    return mask_pil, foreground
