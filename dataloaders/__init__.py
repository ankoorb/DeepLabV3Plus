from dataloaders.dataset import siim, augsiim
from torch.utils.data import DataLoader
from dataloaders.augment import compose_augmentations


def make_data_loader(args, **kwargs):

    if args.dataset == 'siim':
        train_set = siim.SIIMDataset(args, split='train')
        val_set = siim.SIIMDataset(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'augsiim':
        train_set = augsiim.AugSIIMDataset(args, split='train', transforms_fn=compose_augmentations)
        val_set = augsiim.AugSIIMDataset(args, split='val', transforms_fn=compose_augmentations)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

