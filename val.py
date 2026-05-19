import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset.buil_dataset import buil_dataset_distributed, buil_dataset
from dataset.crack_dataset import Crack_Dataset, eval_transform
from src import create_model
from utils.optimizer import build_optim
from utils.train_and_eval import train_one_epoch, evaluate
from utils.utils import init_multi_mode
import os


def main(local_rank, num_gpus,weights_path,data_path,save_name):

    pth_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    args = pth_dict['args']
    args.data_path = data_path

    if num_gpus > 1:
        init_multi_mode(local_rank, num_gpus, args)
        model = create_model(args)
        model.load_state_dict(pth_dict['model'])
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        model.to(device)
        val_dataset = Crack_Dataset(root=args.data_path,
                                    train=False,
                                    transforms=eval_transform(size=args.train_size))
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 sampler=val_sampler,
                                                 pin_memory=True,
                                                 drop_last=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

        model_without_ddp = model.module
    else:
        args.distributed = False
        args.rank = local_rank
        model = create_model(args)
        device = torch.device(args.device)
        model.load_state_dict(pth_dict['model'])
        model.to(device)
        val_dataset = Crack_Dataset(root=args.data_path,
                                    train=False,
                                    transforms=eval_transform(size=args.train_size))
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=num_workers,
                                                 pin_memory=False,
                                                 drop_last=True)
        model_without_ddp = model



    model.to(device)

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = build_optim(args, params_to_optimize)

    confmat, val_loss, dice = evaluate(
        model, optimizer, val_loader, device=device,
        num_classes=args.num_classes, print_freq=args.print_freq
    )
    val_info = str(confmat)
    val_info += f"dice : {dice:.3f}\n"
    print(val_info)
    if args.rank in [-1, 0]:
        with open(save_name, 'w') as f:
            f.write(val_info)


if __name__ == '__main__':

    models = os.listdir('log/')
    for m in models:
        weights_path = f'log/{m}/best_model.pth'
        data_path = 'coco'
        os.makedirs(f'val_results/{m}',exist_ok=True)
        save_name = f'val_results/{m}/val_new.txt'

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for val")
            mp.spawn(
                main,
                args=(num_gpus,weights_path,data_path,save_name),
                nprocs=num_gpus,
                join=True
            )
        else:
            print("Using single GPU for val")
            main(0, 1,weights_path,data_path,save_name)