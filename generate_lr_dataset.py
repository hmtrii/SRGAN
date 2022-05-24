import cv2
import os
from tqdm import tqdm


def gen_lr_cyclegan_dataset(des_root, src_root, factor):
    for subset in tqdm(os.listdir(src_root)):
        for sub_dir_1 in ['train', 'test']:
            for sub_dir_2 in ['A', 'B']:
                tmp_root = os.path.join(src_root, subset, sub_dir_1, sub_dir_2)
                des_image_path = os.path.join(des_root, subset, sub_dir_1, sub_dir_2)
                os.makedirs(des_image_path, exist_ok=True)
                try:
                    for image_name in os.listdir(tmp_root):
                        image = cv2.imread(os.path.join(tmp_root, image_name))
                        new_width, new_height = int(image.shape[1]*factor), int(image.shape[0]*factor)
                        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(os.path.join(des_image_path, image_name), scaled_image)
                except:
                    print(tmp_root)


if __name__ == '__main__':
    gen_lr_cyclegan_dataset('./data/lr_cyclegan', './data/cyclegan', 1/4)