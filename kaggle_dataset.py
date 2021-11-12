import os
import numpy as np
import torch
import json
from PIL import Image


identity_mapping = {
    "ben_afflek": 1,
    "elton_john": 2,
    "jerry_seinfeld": 3,
    "madonna":4,
    "mindy_kaling":5
}



class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pictures"))))
        self.annos = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "pictures", self.imgs[idx])
        anno_path = os.path.join(self.root, "annotations", self.annos[idx])
        img = Image.open(img_path).convert("RGB")

        with open(anno_path) as f:
            annotations = json.load(f)
        
        shapes = annotations["shapes"]

        num_objs = len(shapes)

        boxes = []
        labels = []
        for i in range(num_objs):
            points = shapes[i]["points"]
            xmin = points[0][0]
            xmax = points[1][0]
            ymin = points[0][1]
            ymax = points[1][1]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(identity_mapping[shapes[i]["label"]])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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
        return len(self.imgs)