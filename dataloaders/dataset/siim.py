import os
import collections
import numpy as np
import pandas as pd
from PIL import Image
from datapath import Path
from pydicom import dcmread
from torch.utils import data
from torchvision import transforms
from dataloaders.utils import rle2mask
from dataloaders import custom_transforms as tr



class SIIMDataset(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root=Path.db_root_dir('siim'), split='train'):
        self.args = args
        self.root = root
        self.split = split

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
        sample = {'image': image, 'mask': mask}

        if self.split == 'train':
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def __len__(self):
        return len(self.image_data)

    def read_dicom(self, img_path):
        image = dcmread(img_path).pixel_array

        # Gray scale to RGB PIL
        image = Image.fromarray(image).convert("RGB")

        return image

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

        return Image.fromarray(mask)

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomRotate(degree=10),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.positive_only = False

    siim_train = SIIMDataset(args, split='train')
    print('Number of training images: ', len(siim_train))

    dataloader = DataLoader(siim_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='siim')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

