import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yolov5
from tqdm import tqdm

class CowDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        self.label_paths = glob.glob(os.path.join(root_dir, '*.txt'))
        self.image_paths.sort()
        self.label_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[0] != 500 or image.shape[1] != 300:
            raise ValueError(f"Image at {img_path} has incorrect dimensions: {image.shape}")

        boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls, x_center, y_center, width, height = map(float, parts)
                x_center *= 300
                y_center *= 500
                width *= 300
                height *= 500
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max, int(cls)])

        boxes = np.array(boxes)

        return image, boxes

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in images])
    targets = [{'boxes': torch.tensor(t[:, :4], dtype=torch.float32), 'labels': torch.tensor(t[:, 4], dtype=torch.int64)} for t in targets]
    return images, targets

def main():
    train_dir = './dataset/images/train'
    val_dir = './dataset/images/val'
    test_dir = './dataset/images/test'

    train_dataset = CowDataset(root_dir=train_dir)
    val_dataset = CowDataset(root_dir=val_dir)
    test_dataset = CowDataset(root_dir=test_dir)

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
        epoch_losses = []

        for images, targets in tqdm(train_loader):
            images = images.to('cuda')
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            # Print shapes for debugging
            print(f'Image shapes: {[img.shape for img in images]}')
            print(f'Target shapes: {[{k: v.shape for k, v in t.items()} for t in targets]}')

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_losses.append(losses.item())

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {np.mean(epoch_losses)}')

        model.eval()
        with torch.no_grad():
            val_losses = []
            for images, targets in tqdm(val_loader):
                images = images.to('cuda')
                targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_losses.append(losses.item())

            print(f'Validation Loss: {np.mean(val_losses)}')

        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_results = []
        for images, targets in tqdm(test_loader):
            images = images.to('cuda')
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            results = model(images, targets)
            test_results.append(results)

    print(f'Test Results: {test_results}')
    model.save('./result/yolov5s_cow_final.pt')

if __name__ == "__main__":
    main()

