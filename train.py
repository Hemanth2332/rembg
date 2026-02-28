from src.CONFIG import *
from src.dataloader import BgRemovalDataset
from src.model import Unet, ResNet50_UNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.loss import LossFn

if __name__ == "__main__":

    train_dataset = BgRemovalDataset(
        real_img_path=REAL_IMG_PATH,
        mask_img_path=MASK_IMG_PATH,
        crop_size=512
    ) 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    if MODEL_TYPE == "unet":
        print("Using UNet model...")
        model = Unet(in_ch=IN_CH, out_ch=OUT_CH).to(DEVICE)
    else:
        print("Using ResNet50_UNet model...")
        model = ResNet50_UNet(in_ch=IN_CH, out_ch=OUT_CH).to(DEVICE)

    loss_fn = LossFn()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # load checkpoint if exists

    if LOAD:
        print("Loading checkpoint...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        avg_loss = checkpoint.get("loss", 0.0)

        print(f"Model {MODEL_PATH} Loaded...")
        print("Checkpoint loaded, starting from epoch {}".format(start_epoch))
    else:
        start_epoch = 0
        avg_loss = 0.0
        print("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch,EPOCHS):

        loss_list = []

        model.train()
        loader = tqdm(train_dataloader, desc='Epoch {}/{}'.format(epoch+1, EPOCHS))

        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                pred = model(img)
                loss = loss_fn(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_list.append(loss.detach().item())
            loader.set_postfix(loss=loss.detach().item())

        avg_loss = sum(loss_list) / len(loss_list)
        print("Epoch {} average loss: {}\n".format(epoch+1, avg_loss))

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
            }
            print("Saving checkpoint for epoch {}...".format(epoch+1))
            torch.save(checkpoint, 'models/resnet50_unet_{}.pth'.format(epoch+1))