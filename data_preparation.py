# ########################### 改变测试集需要改变路径#########################

# 读取图片矩阵

from __future__ import print_function
import os
import numpy as np
from skimage.io import imsave, imread


image_rows = 512# 图片尺寸
image_cols = 512# 图片尺寸

# #############################图片预处理################################
def create_train_data(train_data_path, train_dir_star, total):
    #读取训练矩阵  针对train数据    
    images_list=os.listdir(train_data_path)
    dir_num=50# 文件夹数
    max_file=50
    num=0
    imgs=np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask=np.ndarray((total, image_rows,image_cols), dtype=np.uint8)   
    for i in range(dir_num):#文件夹数
        for j in range(max_file):# 文件夹中最大文件数
            try:
                img = imread(train_data_path + str(train_dir_star + i) + '/arterial phase/' + str(10001+j) + '.png')
                img_mask = imread( train_data_path + str(train_dir_star + i) + '/arterial phase/' + str(10001+j) + '_mask.png')

                img = np.array([img])
                img_mask = np.array([img_mask])

                imgs[num] = img
                imgs_mask[num] = img_mask

                if num % 100 == 0:
                    print('Done: {0}/{1} images'.format(num, total))
                num += 1
            except Exception as e:
                break
    print('train数量:'+str(num))
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')
    
    
def create_test_data(test_data_path):# 针对test1数据集    
    images = os.listdir(test_data_path)
    total = len(images)
    print(total)
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    num = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[num] = img
        imgs_id[num] = img_id

        if num % 100 == 0:
            print('Done: {0}/{1} images'.format(num, total))
        num += 1
    print('test数量:' + str(num))
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


if __name__ == '__main__':# 四个文件
  
    train_data_path="keras-unet/train2/"  # 训练文件路径
    train_dir_star = 1051    #1001
    #total = 1365  
    total = 1452
    test_data_path='keras-unet/test1/test1102/Image' # ########################### 改变测试集需要改变路径#########################
    
    create_train_data(train_data_path,train_dir_star,total)# 创建训练图片（原始图片+掩码图片）矩阵（图片数量*图片长*宽）
    create_test_data(test_data_path)# 同上