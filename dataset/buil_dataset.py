import os
import torch
from dataset.crack_dataset import Crack_Dataset, train_transform, eval_transform


def buil_dataset(args):
    train_dataset = Crack_Dataset(root=args.data_path,
                                train=True,
                                transforms=train_transform(size=args.train_size))

    val_dataset = Crack_Dataset(root=args.data_path,
                                train=False,
                                transforms=eval_transform(size=args.train_size))

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             drop_last=True)
    return train_loader,val_loader

def buil_dataset_distributed(args):
    train_dataset = Crack_Dataset(root=args.data_path,
                                  train=True,
                                  transforms=train_transform(size=args.train_size))

    val_dataset = Crack_Dataset(root=args.data_path,
                                train=False,
                                transforms=eval_transform(size=args.train_size))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               drop_last=True
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             drop_last=True)

    return train_loader, val_loader,train_sampler,val_sampler
