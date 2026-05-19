import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import create_model
import torch.nn.functional as F


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(args).to(device)
    model.load_state_dict(pth_dict['model'])
    model.to(device)

    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))

    p = 255 // (args.num_classes - 1)
    model.eval()

    data_transform = transforms.Compose([
        transforms.Resize(args.train_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with torch.no_grad():
        for i in os.listdir(test_path):
            # load image
            img_path = os.path.join(test_path,i)
            original_img = Image.open(img_path).convert('RGB')

            ori_shape = original_img.size #(w,h,c)
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            t_start = time_synchronized()
            output = model(img.to(device))
            if isinstance(output,(list,tuple)):
                output = output[-1]

            output = F.interpolate(output,(ori_shape[1],ori_shape[0]),mode = 'bilinear',align_corners = True)

            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start),' s')
            prediction = output.argmax(1).squeeze(0)

            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction  = prediction * p
            mask = Image.fromarray(prediction)
            mask.save(os.path.join(save_path,i))


if __name__ == '__main__':
    test_path = 'coco/images/val'
    images = os.listdir(test_path)
    models = os.listdir('log/')
    for m in models:
        weights_path = f'log/{m}/best_model.pth'
        pth_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        args = pth_dict['args']
        save_path = f'predictions/{m}'
        main()
