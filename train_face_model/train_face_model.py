import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import WIDERFace
import pandas as pd
from PIL import Image

from FaceRollNet import FaceRollNet

device = "cuda" if torch.cuda.is_available() else "cpu"

class WiderSingleFace(Dataset):

    def __init__(self, root="data", split="train", transform=None):
        self.dataset = WIDERFace(root=root, split=split, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img, target = self.dataset[idx]

        w, h = img.size
        boxes = torch.as_tensor(target["bbox"], dtype=torch.float32)

        if len(boxes) == 0:
            bbox = torch.zeros(4)
            mask_bbox = 0
        else:
            areas = boxes[:, 2] * boxes[:, 3]
            idx_big = areas.argmax()
            x, y, bw, bh = boxes[idx_big]

            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw = bw / w
            bh = bh / h

            bbox = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
            mask_bbox = 1

        if self.transform:
            img = self.transform(img)

        roll_dummy = torch.zeros(1)

        return img, bbox, roll_dummy, mask_bbox, 0
    
class AFLW2000RollDataset(Dataset):

    def __init__(self, root_dir, csv_path, split="trainimg", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        # Leer CSV
        df = pd.read_csv(csv_path)

        # Normalizar roll
        df["roll_norm"] = df["roll"] / 90.0

        # Obtener imágenes reales en carpeta
        available_images = set(os.listdir(self.root_dir))

        # Filtrar solo filas cuyo img_name exista físicamente
        df = df[df["img_name"].isin(available_images)].reset_index(drop=True)
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_path = os.path.join(self.root_dir, row["img_name"])
        image = Image.open(img_path).convert("RGB")

        roll = torch.tensor([row["roll_norm"]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        bbox_dummy = torch.zeros(4, dtype=torch.float32)

        return image, bbox_dummy, roll, 0, 1
    
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

wider_dataset = WiderSingleFace(transform=transform)
aflw2000_dataset = AFLW2000RollDataset(root_dir="data/AFLW2000", csv_path="data/AFLW2000/angle_data.csv", transform=transform)

wider_loader = DataLoader(wider_dataset, batch_size=16, shuffle=True)
roll_loader = DataLoader(aflw2000_dataset, batch_size=16, shuffle=True)

model = FaceRollNet().to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

bbox_loss_fn = nn.SmoothL1Loss()
roll_loss_fn = nn.MSELoss()

EPOCHS = 20

for epoch in tqdm(range(EPOCHS), desc="Epochs"):

    loss_bbox_epoch = 0
    loss_roll_epoch = 0
    count_bbox = 0
    count_roll = 0

    model.train()

    wider_iter = iter(wider_loader)
    roll_iter = iter(roll_loader)

    num_steps = max(len(wider_loader), len(roll_loader))

    step_bar = tqdm(range(num_steps),
                    desc=f"Epoch {epoch+1}",
                    leave=False)

    for step in step_bar:

        # =======================
        # DETECCIÓN (WIDER)
        # =======================
        for param in model.bbox_head.parameters():
            param.requires_grad = True
        for param in model.roll_head.parameters():
            param.requires_grad = False

        if step < len(wider_loader):

            optimizer.zero_grad()

            imgs, bbox, _, _, _ = next(wider_iter)

            imgs = imgs.to(device)
            bbox = bbox.to(device)

            pred_bbox, _ = model(imgs)

            valid = bbox.sum(dim=1) > 0

            if valid.any():
                loss_bbox = bbox_loss_fn(pred_bbox[valid], bbox[valid])
                loss_bbox.backward()
                optimizer.step()

                loss_bbox_epoch += loss_bbox.item()
                count_bbox += 1

        # =======================
        # ROLL (AFLW)
        # =======================
        for param in model.bbox_head.parameters():
            param.requires_grad = False
        for param in model.roll_head.parameters():
            param.requires_grad = True

        if step < len(roll_loader):

            optimizer.zero_grad()

            imgs, _, roll, _, _ = next(roll_iter)

            imgs = imgs.to(device)
            roll = roll.to(device)

            _, pred_roll = model(imgs)

            loss_roll = roll_loss_fn(pred_roll, roll)
            loss_roll.backward()
            optimizer.step()

            loss_roll_epoch += loss_roll.item()
            count_roll += 1

        # Actualizar métricas en barra
        step_bar.set_postfix({
            "det_loss": f"{loss_bbox_epoch/max(count_bbox,1):.4f}",
            "roll_loss": f"{loss_roll_epoch/max(count_roll,1):.4f}"
        })

    print(
        f"\nEpoch {epoch+1} | "
        f"Loss det: {loss_bbox_epoch/max(count_bbox,1):.4f} | "
        f"Loss roll: {loss_roll_epoch/max(count_roll,1):.4f}"
    )

torch.save(model.state_dict(), "face_roll_model_300WLP.pth")