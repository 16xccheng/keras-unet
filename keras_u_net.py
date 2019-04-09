from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K# keras后端   https://keras.io/zh/backend/

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512# 图像长
img_cols = 512# 图像宽

smooth = 1.

def load_train_data():# 读矩阵
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():# 读矩阵
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id
  
# 计算dice指标，判断分割好坏
def dice_coef(y_true, y_pred):# y_true为真实准确值，y_pred为预测值
    y_true_f = K.flatten(y_true)# 捋直
    y_pred_f = K.flatten(y_pred)# 捋直
    # K.sum不加axi（指定方向求和，返回对应方向向量）,则为全元素求和，返回一个数字
    intersection = K.sum(y_true_f * y_pred_f)# 求预测准确的结果（真实准确值和预测值的交集）
    # 原始公式：（2*预测准确值）/（真实准确值+预测值），越大效果越好
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# 损失函数
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# u-net网络模型
def get_unet(pretrained_weights=None):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # optimizer：优化器，如Adam
    # loss:计算损失，dice_coef_loss损失函数
    # metrics: 列表，包含评估模型在训练和测试时的性能的指标，dice_coef
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])# dice_coef_loss损失函数

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def preprocess(imgs):# 矩阵变换
    # np shape函数为对应矩阵维度的**，类似于长宽高，返回元组，这里[0]为对应图片数量
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)# 创建矩阵（对应图片数量*256*256）
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)# 图片缩放：512 -> 256

    imgs_p = imgs_p[..., np.newaxis]# 增加一个维度,3维 -> 4维
    return imgs_p


def train_and_predict():
    print('Loading and preprocessing train data...')
    imgs_train, imgs_mask_train = load_train_data()# 读矩阵（对应图片数量*图片长（512）*图片宽（512））

    imgs_train = preprocess(imgs_train)# 矩阵变换
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # 平均值mean for data centering
    std = np.std(imgs_train)  # 矩阵全局标准差std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # 矩阵归一化scale masks to [0, 1]
    
    # 图片信息矩阵输入并开始训练
    # model = get_unet('unet_weights.h5') #If you are training again, use this line to load the pre-training model
    model = get_unet()

    model_checkpoint = ModelCheckpoint('unet_weights.h5', monitor='val_loss', save_best_only=True)
    print('Fitting model...')
    model.fit(imgs_train, 
              imgs_mask_train, 
              batch_size=16, 
              nb_epoch=200, 
              verbose=1, 
              shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    # 获取测试集信息矩阵
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    model.load_weights('unet_weights.h5')
    # 开始测试
    print('Predicting masks on test data...')

    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    # 保存
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()