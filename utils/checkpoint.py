import torch



def load_checkpoint(ch_fp,model,optimizer,lr_scheduler,scaler,args):
    checkpoint = torch.load(ch_fp, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model.module.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    args.start_epoch = checkpoint['epoch'] + 1 if args.start_epoch == 0 else args.start_epoch
    if args.amp:
        scaler.load_state_dict(checkpoint["scaler"])
    if args.start_epoch == args.epochs:
        print('Num epochs must to  == last num epochs + new num resume epochs!')
    args.best_metric = checkpoint['args'].best_metric
    print('load checkpoint {}'.format(ch_fp))