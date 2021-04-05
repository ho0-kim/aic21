import cv2
import os
import random
import numpy as np
import skimage
import json

class Augmentation():
    def __init__(self, cfg):
        self.cfg = cfg
        self.aug_func = [self.__horizontal_shift__,
                         self.__vertical_shift__,
                         self.__brightness__,
                         self.__horizontal_flip__,
                         self.__add_gausian_noise__,
                         self.__zoom__]
        self.last = len(self.aug_func)-1
        self.crop_size = self.cfg['data']['crop_size']

        assert(self.crop_size[0] == self.crop_size[1]) # if the values are different,
                            # it needs to check if random_augmentation works in a correct way.

    def __horizontal_shift__(self, img):
        ratio = 0.2 # fix to create a list of fuctions
        #assert (ratio >= 0. and ratio < 1)  # Value should be less than 1 and greater than 0

        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = w * ratio
        if ratio > 0:
            img = img[:, :int(w - to_shift), :]
        if ratio < 0:
            img = img[:, int(-1 * to_shift):, :]
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def __vertical_shift__(self, img):
        ratio = 0.2  # fix to create a list of fuctions
        #assert (ratio >= 0. and ratio < 1)  # Value should be less than 1 and greater than 0

        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = h * ratio
        if ratio > 0:
            img = img[:int(h - to_shift), :, :]
        if ratio < 0:
            img = img[int(-1 * to_shift):, :, :]
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def __brightness__(self, img):
        low = 0.9   # fix to create a list of fuctions
        high = 1.1
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def __horizontal_flip__(self, img):
        return cv2.flip(img, 1)

    def __add_gausian_noise__(self, img):
        img = skimage.util.random_noise(img, mode="gaussian")
        img = (img * 255).astype(np.uint8)
        return img

    def __zoom__(self, img):
        value = 0.8 # fix to create a list of fuctions
        #assert (value >= 0 and value < 1)  # Value for zoom should be less than 1 and greater than 0

        value = random.uniform(value, 1)
        h, w = img.shape[:2]
        h_taken = int(value * h)
        w_taken = int(value * w)
        h_start = random.randint(0, h - h_taken)
        w_start = random.randint(0, w - w_taken)
        img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def random_augmentation(self, img, b_all_aug):
        last = self.last
        if img.shape[0] < self.crop_size[0] \
                or img.shape[1] < self.crop_size[1]\
                or b_all_aug == False:
            last -= 2                   # do not apply zoom and noise to small images
        num = random.randint(0, last)
        return self.aug_func[num](img)

def main():
    file_path = '../../validation/S05/c021/img1'
    file_name = '002698.jpg'
    box = [
        1496,
        373,
        422,
        388
    ]

    with open('config.json') as f:
        cfg = json.load(f)

    img = cv2.imread(os.path.join(file_path, file_name))
    crop = img[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]

    aug = Augmentation(cfg)
    img = aug.random_augmentation(crop)

    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()