import os
import time
import datetime
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.buil_dataset import buil_dataset_distributed, buil_dataset
from src import create_model
from utils import init_distributed_mode, save_on_master
import pandas as pd

from utils.checkpoint import load_checkpoint
from utils.optimizer import build_optim
from utils.scheduler import create_lr_scheduler
from utils.train_and_eval import train_one_epoch, evaluate
from utils.utils import init_multi_mode


def main(local_rank, num_gpus, args):


    if num_gpus>1:
        init_multi_mode(local_rank, num_gpus, args)
        model = create_model(args)
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        model.to(device)
        train_loader, val_loader, train_sampler, val_sampler = buil_dataset_distributed(args)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
        model_without_ddp = model.module
    else:
        args.distributed = False
        args.rank = local_rank
        model = create_model(args)
        device = torch.device(args.device)
        model.to(device)
        train_loader, val_loader= buil_dataset(args)
        model_without_ddp = model
        train_sampler, val_sampler = None,None


    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = build_optim(args, params_to_optimize)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_val = ''
    start_time = time.time()

    if args.resume:
        df = pd.read_csv('{}/log.csv'.format(args.log_dir))
        ch_fp = "{}/last_model.pth".format(args.log_dir)
        load_checkpoint(ch_fp=ch_fp, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        scaler=scaler, args=args)
    else:
        df = pd.DataFrame(
            columns=['epoch', 'train_loss', 'val_loss', 'pa', 'miou', 'recall', 'precision', 'dice', 'f1'])

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch,lr_scheduler=lr_scheduler, print_freq=args.print_freq,
            num_classes=args.num_classes, scaler=scaler)

        confmat, val_loss, dice = evaluate(
            model, optimizer, val_loader, device=device,num_classes=args.num_classes, print_freq=args.print_freq)

        val_info = str(confmat)
        val_info += f"train loss : {train_loss:.3f}\n" \
                    f"val loss :{val_loss:.3f}\n" \
                    f"dice : {dice:.3f}\n" \
                    f"epoch : {epoch + 1}\n"  \
                    f"=========================================="
        matrix = confmat.compute()

        if args.rank == 0:
            print(val_info)
            df.loc[epoch] = [
                epoch + 1,
                f'{train_loss:.5f}',
                f'{val_loss:.5f}',
                f'{matrix[0]:.5f}',  # acc
                f'{matrix[2]:.5f}',  # miou
                f'{matrix[3][1:].mean():.5f}',  # recall
                f'{matrix[4][1:].mean():.5f}',  # precision
                f'{dice:.5f}',
                f'{matrix[5][1:].mean():.5f}',
            ]

            df.to_csv(f'{args.log_dir}/log.csv', index=False)

            save_file = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args
            }

            if args.best_metric < round(matrix[2] * 100, 3):
                args.best_metric = round(matrix[2] * 100, 3)
                best_val = val_info
                save_on_master(save_file,"{}/best_model.pth".format(args.log_dir))
                print('save best!', 'miou {}'.format(args.best_metric))
                save_on_master(save_file,"{}/last_model.pth".format(args.log_dir))
                with open(os.path.join(args.log_dir, 'best.txt'), 'w') as f:
                    f.write(best_val)

    total_time = time.time() - start_time
    if args.resume:
        total_time = total_time + args.total_time
    args.total_time = total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.rank == 0:
        with open(os.path.join(args.log_dir, 'best.txt'), 'a+') as f:
            f.write(f"\ntraining time {total_time_str}")
        print(f"training time {total_time_str}")
        print("best :\n", best_val)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="training")

    parser.add_argument("--model",
                        default='crackformer',
                        )
    parser.add_argument("--t1", default=True, type=bool)
    parser.add_argument("--t2", default=False, type=bool)
    parser.add_argument("--t3", default=False, type=bool)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--data_path",
                        default="../coco/")
    parser.add_argument("--device", default="cuda:0", help="training device")
    # device参数已由程序自动设置
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="epochs")
    parser.add_argument("--train_size",
                        default=(512, 512),
                        type=int,
                        help='size(h,w) to training')
    # 学习率
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='log print frequency')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--log_dir", default='log', type=str)
    parser.add_argument("--best_metric", default=0, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    assert len(args.log_dir.split('/')) <= 2
    os.makedirs(args.log_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training")
        mp.spawn(
            main,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True
        )
    else:
        print("Using single GPU for training")
        main(0, 1, args)