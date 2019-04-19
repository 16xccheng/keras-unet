# ########################### 改变训练集需要改变路径#########################
# 读取图片。。。。。。增强对比度，只跑一次。。。。预处理。。。。。。训练集
from PIL import Image #需要下载并安装PIL（python的一个image图像处理库）
from PIL import ImageEnhance
from PIL import ImageFilter

def image_enhance(img):# 对比度增强
    contrast=1.5
    enh_con=ImageEnhance.Contrast(img)
    img_contrasted=enh_con.enhance(contrast)
    return img_contrasted
  
train_data_path_in = 'keras-unet2/train2/'  # 文件路径

train_data_path_out = 'keras-unet/train2/'  # 文件路径
  
total=0
for i in range(50):# 读取train图片。。。。。增强对比度
    for j in range(50):
        infile = train_data_path_in +  str(1051+i) + '/arterial phase/' + str(10001+j) + '.png'
        outfile= train_data_path_out + str(1051+i) + '/arterial phase/' + str(10001+j) + '.png'
        try:
            im = Image.open(infile).convert('L')
            out = im.resize((512,512), Image.ANTIALIAS)
            out = image_enhance(out)# 对比度
            out.save(outfile)
            total+=1
        except Exception as e:
            break
print(total)     



# ########################### 改变测试集需要改变路径#########################

# 读取图片。。。。。。增强对比度+滤波，只跑一次。。。。预处理..............测试集
def image_enhance(img):# 对比度增强
    contrast=1.5
    enh_con=ImageEnhance.Contrast(img)
    img_contrasted=enh_con.enhance(contrast)
    return img_contrasted
  
test_data_path_in = 'keras-unet2/test1/test1102/Image/' # ########### 文件夹路径################

test_data_path_out = 'keras-unet/test1/test1102/Image/' # ########### 文件夹路径################

for i in range(50):# 读取test图片。。。。。增强对比度
    try:
        infile = test_data_path_in +  str(10001+i) + '.png' #原始图像路径
        outfile= test_data_path_out + str(10001+i) + '.png' #灰度化后的图像路径

        
        im = Image.open(infile).convert('L') #灰度化
        out = im.resize((512,512),Image.ANTIALIAS)#重新定义图片尺寸大小
        out = image_enhance(out)# 对比度增强
		
        out= out.filter(ImageFilter.MinFilter(3))# ImageFilter.SHARPEN # ImageFilter.MinFilter(7)# 滤波
        #out= out.filter(ImageFilter.MinFilter(3))
        #out = out.filter(ImageFilter.MinFilter(3))
        #out= out.filter(ImageFilter.MaxFilter(3))
        #out= out.filter(ImageFilter.MaxFilter(3))        
        #out= out.filter(ImageFilter.MaxFilter(3))
        #out= out.filter(ImageFilter.MinFilter(3))
        out.save(outfile)
    except Exception as e:
        break