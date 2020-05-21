import os
import numpy as np
import pandas as pd
import ast
import torch
from PIL import Image
import matplotlib.pyplot as plt


class WheatDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # Attention: some images may not be listed in the csv data as they
        # have no wheat heads. This is why self.imgs is listed from the dir and
        # not the data
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))

        # data:
        df = pd.read_csv(os.path.join(root, "train.csv"),
                        converters={"bbox":ast.literal_eval})
        self.data = df.groupby("image_id").apply(self.getinfo)

        self.imgs_with_heads = set(self.data.index)
        #self.imgs_without_heads = set(img[:-4] for img in self.imgs if img[:-4] not in self.imgs_with_heads)

    def getinfo(img_id, image_info):
        #height = image_info.iloc[0]["height"]
        #width = image_info.iloc[0]["width"]
        box_count = image_info.shape[0]
        bboxes = np.array(image_info["bbox"].to_list())
        bboxes[:,2] += bboxes[:,0]
        bboxes[:,3] += bboxes[:,1]

        # If func returns a Series object the result will be a DataFrame.
        return pd.Series([box_count, bboxes],
                         index=['box_count', 'bboxes'])

    def __getitem__(self, idx):
        # Attention: some images may not be listed in the csv data if they
        # have no wheat heads
        # load images
        img_filename = self.imgs[idx]
        # remove extension for img id
        """m = re.match("(\w+).[jpg|png]", img_filename)
        img_id = m.group(1)
        img_id = img_filename.split('.')[0]"""
        img_id = img_filename[:-4]

        img_path = os.path.join(self.root, "train", img_filename)
        img = Image.open(img_path).convert("RGB")

        if img_id in self.imgs_with_heads:
            img_info = self.data.loc[img_id]

            #height = img_info["height"]
            #width = img_info["width"]
            box_count = img_info["box_count"]
            bboxes = img_info["bboxes"]
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        else:
            #height = img.height
            #width = img.width
            box_count = 0
            bboxes = np.array([]).reshape(-1, 4)
            area = np.array([])

        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((box_count,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        #height = torch.tensor([height], dtype=torch.float32)
        #width = torch.tensor([width], dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((box_count,), dtype=torch.int64)

        target = {}
        #target["height"] = height
        #target["width"] = width
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
