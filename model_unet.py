############################ 改变模型数据写入需要改变路径#########################
#训练并测试
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

img_rows = 256# 图像长
img_cols = 256# 图像宽

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

# ################################################ 原始u-net 网络模型  #############################################
def get_unet(pretrained_weights=None):
    print('使用的是 原始get_unet')
    weight=32
    nb_filter = [weight, weight*2, weight*4, weight*8, weight*16]
    
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # optimizer：优化器，如Adam
    # loss:计算损失，dice_coef_loss损失函数
    # metrics: 列表，包含评估模型在训练和测试时的性能的指标，dice_coef
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])# dice_coef_loss损失函数

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
  
  
# ################################################ 原始u-net wide 网络模型  #############################################
def get_unetw(pretrained_weights=None):
    print('使用的是 原始get_unet wide')
    
    weight=38
    nb_filter = [weight,weight*2,weight*4,weight*8,weight*16]
    
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # optimizer：优化器，如Adam
    # loss:计算损失，dice_coef_loss损失函数
    # metrics: 列表，包含评估模型在训练和测试时的性能的指标，dice_coef
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])# dice_coef_loss损失函数

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# ################################################ 原始u-net++网络模型  #############################################

def get_unetpp(num_class=1, deep_supervision=False):
    print('使用的是原始unet++')
    nb_filter = [32,64,128,256,512]
    img_rows=256
    img_cols=256
    color_type=1
    bn_axis = 3
    
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')

    
    conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(img_input)
    conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    
    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    
    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1) 
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_2)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_2)    

    
    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    
    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_2)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_2)

    
    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_3)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_3)

    
    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    
    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3_2)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3_2)

    
    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_3)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_3)

    
    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_4)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_4)

    
    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5_1)

    
    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4_2)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4_2)

    
    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3_3)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3_3)

    
    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_4)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2_4)

    
    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_5)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1_5)

    
    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    
    model = Model(input=img_input, output=[nestnet_output_4])
    
    # optimizer：优化器，如Adam
    # loss:计算损失，dice_coef_loss损失函数
    # metrics: 列表，包含评估模型在训练和测试时的性能的指标，dice_coef
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])# dice_coef_loss损失函数


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
    #model = get_unet('/opt/bin/unet_weights.h5') #重复训练则读取本文件继续加载训练
    
    #model = get_unet()  # 原始unet
    model = get_unetw()  # 原始unetw
    #model = get_unetpp() # 原始unet++
    
    
    # 图片保存路径
    #pred_dir = 'preds_unet'
    pred_dir = 'preds_unetw_38'
    #pred_dir = 'preds_unetpp'
    
    #save_res = '/opt/bin/unet_weights.h5'############################ 改变模型数据写入需要改变路径#########################
    save_res = '/opt/bin/unet_weightsw_38.h5'
    #save_res = '/opt/bin/unet_weightspp.h5'
    
    
    #''' 
	#在训练生成 .h5 文件后可以直接注释该段选用其他测试集进行测试
    model_checkpoint = ModelCheckpoint(save_res, monitor='val_loss', save_best_only=True)
    print('Fitting model...')
    model.fit(imgs_train, 
              imgs_mask_train, 
              batch_size=16, 
              nb_epoch=50, # 训练次数
              verbose=1, 
              shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    #'''
    
    
    # 获取测试集信息矩阵
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    model.load_weights(save_res)
    # 开始测试
    print('Predicting masks on test data...')

    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    
    
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_mask.png'), image)

if __name__ == '__main__':
    train_and_predict()