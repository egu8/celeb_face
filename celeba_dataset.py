import os
import numpy as np
import torch
import json
from PIL import Image

class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_path, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.annos = list(sorted(os.listdir(root)))
        self.img_path = img_path

    def __getitem__(self, idx):
        # load images and masks

        anno_path = os.path.join(self.root, self.annos[idx])

        with open(anno_path) as f:
            annotations = json.load(f)
        
        box = annotations["box"]
        label = annotations["label"]
        img_path = annotations["path"]

        img_path = os.path.join(self.img_path, img_path)
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for i in range(1):
            x,y, w, h = box
            
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annos)