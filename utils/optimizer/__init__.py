import torch


def build_optim(args,params_to_optimize):
    optimizer = torch.optim.SGD(params_to_optimize,
                                    momentum=0.9,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    return optimizer