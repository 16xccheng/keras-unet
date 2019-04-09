from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = 'keras-unet/data/'# 基本路径

image_rows = 512# 图片尺寸
image_cols = 512# 图片尺寸


def create_train_data():
    train_data_path = os.path.join(data_path, 'train/Image')# 原始图集
    train_data_Label_path = os.path.join(data_path, 'train/Label')# 掩码
    images = os.listdir(train_data_path)
    total = len(images)# 训练图集数量

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        img      = imread(os.path.join(train_data_path, image_name), as_grey=True)# 修改图片名可于此修改
        img_mask = imread(os.path.join(train_data_Label_path, image_name), as_grey=True)# 修改图片名可于此修改
        # 原始图片和掩码图片矩阵
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test/Image')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':# 四个文件
    create_train_data()# 创建训练图片（原始图片+掩码图片）矩阵（图片数量*图片长*宽）
    create_test_data()# 同上