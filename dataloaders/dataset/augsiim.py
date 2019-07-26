import os
import collections
import numpy as np
import pandas as pd
from PIL import Image
from datapath import Path
from pydicom import dcmread
from torch.utils import data
from dataloaders.utils import rle2mask
from dataloaders.augment import compose_augmentations


class AugSIIMDataset(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root=Path.db_root_dir('siim'), split='train', transforms_fn=compose_augmentations):
        self.args = args
        self.root = root
        self.split = split
        self.transforms_fn = transforms_fn

        if split == 'train':
            header_csv_path = os.path.join(self.root, 'train-split-headers.csv')
            rle_csv_path = os.path.join(self.root, 'train-split-rle.csv')
        elif split == 'val':
            header_csv_path = os.path.join(self.root, 'val-split-headers.csv')
            rle_csv_path = os.path.join(self.root, 'val-split-rle.csv')

        self.df = pd.read_csv(header_csv_path)
        self.df.columns = list(map(lambda x: x.strip(), self.df.columns))
        self.rle_df = pd.read_csv(rle_csv_path)
        self.rle_df.columns = list(map(lambda x: x.strip(), self.rle_df.columns))

        # Use positive samples only
        if self.args.positive_only:
            self.rle_df = self.rle_df[self.rle_df['EncodedPixels'] != ' -1']
            self.rle_df.reset_index(inplace=True, drop=True)
            positive_ids = self.rle_df['ImageId'].unique().tolist()
            self.df = self.df[self.df['ImageId'].isin(positive_ids)]
            self.df.reset_index(inplace=True, drop=True)

        self.height = 1024
        self.width = 1024
        self.image_data = collections.defaultdict(dict)

        counter = 0
        for index, row in self.df.iterrows():
            image_id = row['ImageId']
            filename = row['filename']
            image_path = os.path.join(self.root, filename)
            self.image_data[counter]["image_id"] = image_id
            self.image_data[counter]["image_path"] = image_path
            counter += 1

    def __getitem__(self, index):
        image_id = self.image_data[index]["image_id"]
        image_path = self.image_data[index]["image_path"]
        image = self.read_dicom(image_path)
        mask = self.generate_mask(image_id)

        if self.transforms_fn:
            augmented = self.transforms(image, mask)

        sample = {'image': augmented['image'],
                  'mask': augmented['mask'] * 255} # Multiply 255 because albumentations ToTensor divides by 255

        return sample

    def __len__(self):
        return len(self.image_data)

    def read_dicom(self, img_path):
        image = dcmread(img_path).pixel_array

        # Gray scale to RGB PIL
        image = Image.fromarray(image).convert("RGB")

        return np.array(image)

    def generate_mask(self, image_id):
        rle_data = self.rle_df.query("ImageId=='{}'".format(image_id))['EncodedPixels'].tolist()

        n_rle = len(rle_data)

        # Blank Mask
        if n_rle == 1 and rle_data[0].strip() == '-1':
            mask = np.zeros((self.width, self.height), dtype=np.uint8)
        # Mask
        else:
            mask = np.zeros((self.width, self.height), dtype=np.uint8)
            for i, rle in enumerate(rle_data):
                temp_mask = rle2mask(rle, self.width, self.height).T.astype(np.uint8)
                mask |= (temp_mask > 0)  # Merge temp masks to a single mask

        return mask

    def transforms(self, image, mask):
        composed_transforms = self.transforms_fn(img_height=self.args.base_size,
                                                 img_width=self.args.base_size,
                                                 flip_p=self.args.flip_prob,
                                                 translate_p=self.args.translate_prob,
                                                 distort_p=self.args.distort_prob,
                                                 color_p=self.args.color_prob,
                                                 overlays_p=self.args.overlays_prob,
                                                 blur_p=self.args.blur_prob,
                                                 noise_p=self.args.noise_prob)

        return composed_transforms(image=image, mask=mask)

