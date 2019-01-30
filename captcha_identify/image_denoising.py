from PIL import Image
from sklearn import preprocessing
import numpy as np
import pandas as pd
import os, time, asyncio
from multiprocessing import Pool
captcha_path = 'data/captcha'
captcha_binary_path = 'data/captcha_binary'
captcha_denoising_path = 'data/captcha_denoising'


def image_denoising( path, file_list):

    loop = asyncio.get_event_loop()
    tasks = []

    for cur_file in file_list:
        # image_denoising_detail(path, cur_file)
        tasks.append( image_denoising_detail(path, cur_file) )
    loop.run_until_complete( asyncio.wait( tasks ) )


async def image_denoising_detail( path, cur_file ):

    img = Image.open(path + '/' + cur_file).convert('L')
    binary_data = np.array(img, dtype='int')
    copy_binary_data = binary_data.copy()
    img.close()
    denosing(binary_data)
    # print( (copy_binary_data == binary_data).all() )
    binary_data = binary_data.astype('uint8')
    img = Image.fromarray(binary_data)
    with open(captcha_denoising_path + '/' + cur_file, 'wb') as f:
        img.save(f)
    img.close()




def denosing( binary_data ):
    rows, cols = binary_data.shape[:2]
    for r in range(0, rows):
        for c in range(0, cols):
            cur = binary_data[r, c]
            if r == 0:  #
                if c == 0:  # 最左上角
                    sum = cur + \
                          binary_data[r + 1, c] + \
                          binary_data[r, c + 1] + \
                          binary_data[r + 1, c + 1]
                    if sum - 200 * 2 > 0:
                        binary_data[r, c] = 255
                elif c == cols - 1:  # 最右上角
                    sum = cur + \
                          binary_data[r, c - 1] + \
                          binary_data[r + 1, c] + \
                          binary_data[r + 1, c - 1]
                    if sum - 200 * 2 > 0:
                        binary_data[r, c] = 255
                else:  # 上边缘点
                    sum = cur + \
                          binary_data[r, c - 1] + \
                          binary_data[r, c + 1] + \
                          binary_data[r + 1, c - 1] + \
                          binary_data[r + 1, c] + \
                          binary_data[r + 1, c + 1]
                    if sum - 200 * 4 > 0:
                        binary_data[r, c] = 255
            elif r == rows - 1:
                if c == 0:  # 最左下角
                    sum = cur + \
                          binary_data[r, c + 1] + \
                          binary_data[r - 1, c] + \
                          binary_data[r - 1, c + 1]
                    if sum - 200 * 2 > 0:
                        binary_data[r, c] = 255
                elif c == cols - 1:  # 右下角
                    sum = cur + \
                          binary_data[r, c - 1] + \
                          binary_data[r - 1, c] + \
                          binary_data[r - 1, c - 1]
                    if sum - 200 * 2 > 0:
                        binary_data[r, c] = 255
                else:  # 下边缘点
                    sum = cur + \
                          binary_data[r, c - 1] + \
                          binary_data[r, c + 1] + \
                          binary_data[r - 1, c - 1] + \
                          binary_data[r - 1, c] + \
                          binary_data[r - 1, c + 1]
                    if sum - 200 * 4 > 0:
                        binary_data[r, c] = 255

            else:
                if c == 0:  # 左边缘点
                    sum = cur + \
                          binary_data[r - 1, c] + \
                          binary_data[r + 1, c] + \
                          binary_data[r - 1, c + 1] + \
                          binary_data[r, c + 1] + \
                          binary_data[r + 1, c + 1]
                    if sum - 200 * 4 > 0:
                        binary_data[r, c] = 255
                elif c == cols - 1:  # 右边缘点
                    sum = cur + \
                          binary_data[r - 1, c] + \
                          binary_data[r + 1, c] + \
                          binary_data[r - 1, c - 1] + \
                          binary_data[r, c - 1] + \
                          binary_data[r + 1, c - 1]
                    if sum - 200 * 4 > 0:
                        binary_data[r, c] = 255
                else:  # 中间点
                    sum = cur + \
                          binary_data[r - 1, c - 1] + \
                          binary_data[r, c - 1] + \
                          binary_data[r + 1, c - 1] + \
                          binary_data[r - 1, c] + \
                          binary_data[r + 1, c] + \
                          binary_data[r - 1, c + 1] + \
                          binary_data[r, c + 1] + \
                          binary_data[r + 1, c + 1]

                    if sum - 240 * 7 > 0:
                        binary_data[r, c] = 255






if __name__ == '__main__':
    start_time = time.time()
    p = Pool()
    file_list = np.array(os.listdir( captcha_binary_path ))
    file_list_4 = np.split(file_list, 4, axis=0)
    resultArr = []
    for cur_file_list in file_list_4:
        # image_denoising(captcha_binary_path, cur_file_list)
        resultArr.append(p.apply_async( image_denoising, args=( captcha_binary_path, cur_file_list ) ))
    p.close()
    print( len(file_list) )
    for i in resultArr:
        i.get()
    end_time = time.time()
    print('图片二值化完毕')
    print('消耗时间：' + str(end_time - start_time))
    # image_binary( '0ABY.jpg' )


