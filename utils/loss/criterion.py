
import torch
from torch import nn
from utils import utils
from utils.loss.dice_coefficient_loss import build_target,dice_loss
from utils.utils import DiceCoefficient


def criterion(inputs, target,loss_weight=None,ignore_index: int = -100, num_classes = 2):

    if isinstance(inputs,list):
        loss = 0.0
        for input in inputs:
            loss_ = nn.functional.cross_entropy(input, target, ignore_index = ignore_index, weight = loss_weight)
            dice_target = build_target(target, num_classes, ignore_index)
            loss_ = loss_ + dice_loss(input, dice_target, ignore_index = 255)
            loss+=loss_
    else:
        loss = nn.functional.cross_entropy(inputs, target, ignore_index = ignore_index, weight = loss_weight)
        dice_target = build_target(target, num_classes, ignore_index)
        loss  = loss +dice_loss(inputs, dice_target,ignore_index = 255)
    return loss