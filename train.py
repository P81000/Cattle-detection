#!/usr/bin/env python3

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import yolov5

class CowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(root_dir, '*.txt')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls, x_center, y_center, box_width, box_height = map(float, parts)
                x_center /= width
                y_center /= height
                box_width /= width
                box_height /= height
                boxes.append([x_center, y_center, box_width, box_height, int(cls)])

        boxes = np.array(boxes)

        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes[:, :4], class_labels=boxes[:, 4])
            image = augmented['image']
            boxes = np.concatenate([augmented['bboxes'], np.expand_dims(boxes[:, 4], axis=1)], axis=1)

        target = {}
        target['boxes'] = torch.as_tensor(boxes[:, :4], dtype=torch.float32)
        target['labels'] = torch.as_tensor(boxes[:, 4], dtype=torch.int64)

        return image, target

transform = A.Compose(
    [
        A.Resize(300, 300),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

train_dir = './dataset/images/train'
val_dir = './dataset/images/val'
test_dir = './dataset/images/val'

train_dataset = CowDataset(root_dir=train_dir, transform=transform)
val_dataset = CowDataset(root_dir=val_dir, transform=transform)
test_dataset = CowDataset(root_dir=test_dir, transform=transform)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

model = yolov5.load('yolov5s.pt')

model.names = ['Cow']

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

epochs = 100
for epoch in range(epochs):
    model.train()

    for images, targets in tqdm(train_loader):
        images = images.to('cuda')
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to('cuda')
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {losses.item()}')

    scheduler.step()

model.eval()
with torch.no_grad():
    for images, targets in tqdm(test_loader):
        images = images.to('cuda')
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        results = model(images, targets)

print(results)

model.save('./result/yolov5s_cow_final.pt')

