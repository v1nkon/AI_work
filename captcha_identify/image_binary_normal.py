from multiprocessing import Pool
import numpy as np
import time
from PIL import Image
import os


captcha_path = 'data/captcha'
captcha_binary_path = 'data/captcha_binary_normal'

#多进程
def read_captcha(path):
    image_array = []
    image_label = []

    p = Pool()
    file_list = np.array( os.listdir(path) )  # 获取captcha文件
    file_list_10 = np.split( file_list, 4, axis=0 )
    file_arr = []
    tasks = []
    start = time.time()

    for file_list_cur in file_list_10:
        p.apply_async(func=read_captcha_detail, args= ( path, file_list_cur) )
        # p.apply_async(func=read_captcha_detail2, args=(path, file_list_cur))
    p.close()
    p.join()
    end = time.time()
    print('时间消耗')

    print(end - start)
    # print( result )
    return file_arr,2

def read_captcha_detail(path, file_list):
    """
    读取验证码图片
    :param path: 原始验证码存放路径
    :return: image_array, image_label：存放读取的iamge list和label list
    """
    image_array = []
    image_label = []
    for file in file_list:
        image = pil_img2numpy_arr( path + '/' + file )
        file_name = file.split(".")[0]
        image_array.append(image)
        image_label.append(file_name)

    image_transfer( image_array, image_label, captcha_binary_path )

    # return image_array, image_label


def image_transfer(image_array, image_label, captcha_binary_path, captcha_clean_save = True):
    """
    图像粗清理
    将图像转换为灰度图像，将像素值小于某个值的点改成白色
    :param image_arry:
    :param captcha_clean_save:
    :return: image_clean:清理过后的图像list
    """
    image_clean = []
    threshold_grey =  140
    for i, image_arr in enumerate(image_array):
        image = numpy_arr2pil_img( image_arr )
        image = image.convert('L') #转换为灰度图像，即RGB通道从3变为1
        im2 = Image.new("L", image.size, 255)

        for y in range(image.size[1]): #遍历所有像素，将灰度超过阈值的像素转变为255（白）
            for x in range(image.size[0]):
                pix = image.getpixel((x, y))
                if int(pix) > threshold_grey:  #灰度阈值
                    im2.putpixel((x, y), 255)
                else:
                    im2.putpixel((x, y), 0)

        if captcha_clean_save: #保存清理过后的iamge到文件
            im2.save(captcha_binary_path + '/' + image_label[i] + '.jpg')
        image_clean.append(im2)
    return image_clean

def pil_img2numpy_arr( img_path ):
    with open( img_path, 'rb' ) as f:
        pil_image = Image.open(f)
        pil_arr = np.array( pil_image )
        return pil_arr
def numpy_arr2pil_img( img_arr ):
    return Image.fromarray(img_arr)

if __name__ == '__main__':

    read_captcha( captcha_path )

