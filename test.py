import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
from PIL import Image
from pydicom import dcmread

from datapath import Path
from dataloaders.utils import mask2rle
from dataloaders import custom_transforms as tr
from dataloaders.tta import TTA, get_tta_function
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *

import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader


class SIIMTestSet(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root=Path.db_root_dir('siim'), split='test'):
        self.args = args
        self.root = root
        self.split = split
        header_csv_path = os.path.join(self.root, 'test-stage1-headers.csv')
        self.df = pd.read_csv(header_csv_path)
        self.df.columns = list(map(lambda x: x.strip(), self.df.columns))
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
        return self.transform_test(sample), image_id

    def __len__(self):
        return len(self.image_data)

    def read_dicom(self, img_path):
        image = dcmread(img_path).pixel_array

        # Gray scale to RGB PIL
        image = Image.fromarray(image).convert("RGB")

        return image

    def generate_mask(self, image_id):
        # Mask with zeros
        mask = np.zeros((self.width, self.height), dtype=np.uint8)

        return Image.fromarray(mask)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


def make_data_loader(args, **kwargs):
    if args.dataset == 'siim':
        test_set = SIIMTestSet(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return test_loader, num_class
    else:
        raise NotImplementedError


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define Metwork
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        self.model = model

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Load checkpoint
        if args.checkpoint_path is not None:
            if not os.path.isfile(args.checkpoint_path):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.checkpoint_path))

            # Threshold
            if self.args.threshold:
                self.threshold = self.args.threshold
            else:
                self.threshold = checkpoint['best_thresh']
            print("Using Threshold: {}".format(self.threshold))

        # TTA function and object
        self.tta_function = get_tta_function(self.args)
        self.tta = TTA(model=self.model, tta_function=self.tta_function)

    # # Sigmoid/Softmax
    def test(self):

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        pred_list = []
        for i, (sample, id) in enumerate(tbar):
            image, image_id = sample['image'], id[0]  # id is a list!
            if self.args.cuda:
                image = image.cuda()

            with torch.no_grad():
                if self.args.tta_fn:
                    output = self.tta(image)
                else:
                    output = self.model(image)

            if self.args.use_sigmoid:
                prob = torch.sigmoid(output[:, 1, :, :])
                pred = prob[0].cpu().numpy()
            else:
                prob = torch.softmax(output, dim=1)
                pred = prob[:, 1, ...].squeeze().cpu().numpy()

            # Compute mask
            mask = (pred > self.threshold).astype(np.uint8).T

            if np.count_nonzero(mask) == 0:
                rle = " -1"
            else:
                rle = mask2rle((mask * 255).astype(np.uint8), self.args.base_size, self.args.base_size)

            pred_list.append([image_id, rle])

        submission_df = pd.DataFrame(pred_list, columns=['ImageId', 'EncodedPixels'])
        submission_df.to_csv(self.args.submission_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3+ Inference")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='siim',
                        choices=['siim'],
                        help='dataset name (default: siim)')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='path to best checkpoint')
    parser.add_argument('--submission-path', type=str, default=None,
                        help='path to save submission csv')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: 1)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold (default=None)')
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    # prediction
    parser.add_argument('--use-sigmoid', type=bool, default=False,
                        help='whether to sigmoid or softmax (default: False)')
    parser.add_argument('--tta-fn', type=str, default=None,
                        choices=['dihedral', 'nonspatial', 'combined'],
                        help='Test Time Augmentation function')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 1

    print(args)
    tester = Tester(args)
    print('--- Inference Started ---')
    tester.test()
    print('--- Inference Finished ---')
    print('Submission file saved at: {}'.format(args.submission_path))


if __name__ == "__main__":
    main()
