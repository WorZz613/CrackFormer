import torch
from torch import nn
from utils import utils
from utils.loss.criterion import criterion
from utils.utils import DiceCoefficient
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model,optimizer, data_loader, device, num_classes,print_freq = 10):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    dice = DiceCoefficient(num_classes = num_classes, ignore_index = 255)
    header = 'Val:'
    with torch.no_grad():
        for image,target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target,ignore_index=255,num_classes = num_classes)

            if isinstance(output,list):
                output = output[-1]
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.detach().item(), lr=lr)
        confmat.reduce_from_all_processes()
    loss = metric_logger.meters["loss"].global_avg
    return confmat,loss,dice.value.item()

def evaluate_roc(model,optimizer, data_loader, device, num_classes,print_freq = 10):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    dice = DiceCoefficient(num_classes = num_classes, ignore_index = 255)
    header = 'Val:'

    all_pred_probs = []
    all_gt_labels = []
    class_names = [f"Class {i}" for i in range(num_classes)]  # 类别名称


    with torch.no_grad():
        for image,target ,in metric_logger.log_every(data_loader, print_freq, header):
            image, target= image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target,ignore_index=255,num_classes = num_classes)
            if isinstance(output, (list, tuple)):
                output = output[0]
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
            dice.reduce_from_all_processes()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.detach().item(), lr=lr)

            probs = F.softmax(output, dim=1)

            mask = (target != 255)[0]
            valid_probs = probs[0][:, mask].cpu().numpy()  # [C, valid_pixels]
            valid_labels = target[0][mask].cpu().numpy()  # [valid_pixels]
            all_pred_probs.append(valid_probs)
            all_gt_labels.append(valid_labels)
        confmat.reduce_from_all_processes()

        all_pred_probs = np.concatenate(all_pred_probs, axis=1).T  # [N, C]
        all_gt_labels = np.concatenate(all_gt_labels)  # [N]

        fpr, tpr, roc_auc = {}, {}, {}
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'purple', 'orange']  # 不同类别的颜色
        for i in range(num_classes):
            # 二值化当前类别的标签
            binary_labels = (all_gt_labels == i).astype(np.int32)
            fpr[i], tpr[i], _ = roc_curve(binary_labels, all_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        # 绘制随机猜测线
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC=0.5)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.savefig('../roc_curve.png', dpi=300, bbox_inches='tight')
        try:
            plt.show()
        except:
            pass
    loss = metric_logger.meters["loss"].global_avg
    return confmat, loss, dice.value.item(), roc_auc



def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    lr_scheduler,print_freq=10,num_classes = 2,scaler = None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch+1)

    for image,target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast('cuda',enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, ignore_index=255, num_classes=num_classes)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.detach().item(), lr=lr)
    loss = metric_logger.meters["loss"].global_avg
    return loss, lr

