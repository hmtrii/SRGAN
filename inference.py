import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms as T

from model import Generator


def infer_one_image(image_path, generator):
    lr_image = Image.open(image_path)
    lr_image_tensor = T.ToTensor()(lr_image).unsqueeze_(0)
    lr_image_tensor = lr_image_tensor.to(device)

    with torch.no_grad():
        sr_image_tensor = generator(lr_image_tensor)
    sr_image = T.ToPILImage()(sr_image_tensor.squeeze_(0))
    return sr_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Using the SRGAN model generator super-resolution images.")
    parser.add_argument('--input_path', '-i', help='Path to input image file or directory')
    parser.add_argument('--output_path', '-o', help='Path to save directory', default='./inference')
    parser.add_argument('--weight', '-w', help='Path to model weight')
    parser.add_argument('--device', '-d', help='cpu or cuda', default='cpu')
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    weight_path = args.weight
    device = torch.device(args.device)
    os.makedirs(output_path, exist_ok=True)

    checkpoint = torch.load(weight_path, map_location=device)
    generator = Generator().to(device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()

    if os.path.isfile(input_path):
        image_name = os.path.basename(input_path)
        sr_image = infer_one_image(input_path, generator)
        save_path = os.path.join(output_path, image_name)
        sr_image.save(save_path)
        print(f'Output image is save at {save_path}')
    else:
        dir_name = os.path.basename(input_path)
        save_dir = os.path.join(output_path, dir_name)
        os.makedirs(save_dir, exist_ok=True)
        for image_name in tqdm(os.listdir(input_path)):
            sr_image = infer_one_image(os.path.join(input_path, image_name), generator)
            save_path = os.path.join(save_dir, image_name)
            sr_image.save(save_path)
