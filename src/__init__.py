
from src.crackformer.crackformer import CrackFormer


def create_model(args):
    if args.model == 'crackformer':
        model = CrackFormer(num_classes=args.num_classes,
                               pretrained=True, mit='b1', t1=args.t1, t2=args.t2, t3=args.t3)
    else: 
        ValueError('model name!')
    return model

