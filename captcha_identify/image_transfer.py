from PIL import Image
from sklearn import preprocessing
import numpy as np
import os, time, asyncio
from multiprocessing import Pool
captcha_path = 'data/captcha'
captcha_binary_path = 'data/captcha_binary'


def image_transfer( path, file_list):

    loop = asyncio.get_event_loop()
    tasks = []

    for cur_file in file_list:
        # image_denoising_detail(path, cur_file)
        tasks.append( image_transfer_detail(path, cur_file) )
    loop.run_until_complete( asyncio.wait( tasks ) )


async def image_transfer_detail( path, cur_file ):

    img = Image.open(path + '/' + cur_file).convert('L')
    binary_data = np.array(preprocessing.binarize(img, threshold=100))
    img.close()
    binary_data[ binary_data == 1 ] = 255
    img = Image.fromarray(binary_data)
    with open(captcha_binary_path + '/' + cur_file, 'wb') as f:
        img.save(f)
    img.close()







if __name__ == '__main__':
    start_time = time.time()
    p = Pool()
    file_list = np.array(os.listdir( captcha_path ))
    file_list_4 = np.split(file_list, 4, axis=0)
    resultArr = []
    for cur_file_list in file_list_4:
        # image_denoising(captcha_binary_path, cur_file_list)
        resultArr.append(p.apply_async( image_transfer, args=( captcha_path, cur_file_list ) ))
    p.close()
    p.join()
    print( len(file_list) )
    for i in resultArr:
        i.get()
    end_time = time.time()
    print('图片二值化完毕')
    print('消耗时间：' + str(end_time - start_time))
    # image_binary( '0ABY.jpg' )


