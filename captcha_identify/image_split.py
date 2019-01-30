from PIL import Image
from sklearn import preprocessing
import numpy as np
import os, time, asyncio
from multiprocessing import Pool
import logging,sys

logger = logging.getLogger('vinkon')
formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(lineno)s][%(levelname)s][%(message)s]')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('log/log.txt')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


captcha_denoising_path = 'data/captcha_denoising'
captcha_split_path = 'data/captcha_split'
captcha_split_type_path = 'data/captcha_split_type'
captcha_error_path = 'data/captcha_error'
text_num = 4


def make_type_dir( isExist = True ):
    text_type = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if not isExist:
        for type in text_type:
            os.makedirs( captcha_split_type_path + '/' + type )

def image_split( path, file_list):

    loop = asyncio.get_event_loop()
    tasks = []

    for cur_file in file_list:
        # image_denoising_detail(path, cur_file)
        tasks.append( image_split_detail(path, cur_file) )
    loop.run_until_complete( asyncio.wait( tasks ) )


async def image_split_detail( path, cur_file ):

    img = Image.open(path + '/' + cur_file).convert('L')
    binary_data = np.array(preprocessing.binarize(img, threshold=140))
    img.close()
    binary_data[binary_data == 1] = 255
    split_image( binary_data, cur_file )

def split_image( binary_data, cur_file ):
    try:
        file_name = cur_file.split('.')[0]
        rows, cols = binary_data.shape[:2]
        split_line = []
        is_start = True
        is_end = False
        for col in range(0, cols):
            for row in range(0, rows):
                if binary_data[row,col] == 0:
                    if is_start:
                        split_line.append( col )
                        is_start = False
                        is_end = True
                    break;
                if binary_data[row, col] == 255 and row == rows -1 and is_end:
                    split_line.append( col )
                    is_start = True
                    is_end = False
        split_images = []
        text_length = len(split_line) / 2
        for start,end in np.split(np.array( split_line ), text_length, axis=0):
            length = int( end - start )
            if length < 6:
                continue;
            if text_length < text_num and length > 17:
                center = start + int( length / 2 )
                start1 = start
                end1 = center
                start2 = center + 1
                end2 = end
                split_images.append(Image.fromarray(binary_data[:, start1:end1]))
                split_images.append(Image.fromarray(binary_data[:, start2:end2]))
                continue;
            split_images.append( Image.fromarray(binary_data[:,start:end]) )
        for index,img in enumerate(split_images):
            dir_type = file_name[index]
            file_path_name = captcha_split_type_path + '/' + dir_type +'/' + file_name + '_' + dir_type + '.jpg'
            with open(file_path_name, 'wb') as f:
                img.save(f)
    except Exception:
        print( cur_file )
        print(sys.exc_info())
        with open( captcha_error_path + '/' +  cur_file , 'wb' )as f:
            Image.fromarray(binary_data).save(f)
        logger.error(sys.exc_info())





if __name__ == '__main__':
    start_time = time.time()
    p = Pool()
    file_list = np.array(os.listdir( captcha_denoising_path ))
    file_list_4 = np.split(file_list, 4, axis=0)
    resultArr = []
    make_type_dir(  False )
    for cur_file_list in file_list_4:
        image_split(captcha_denoising_path, cur_file_list)
        # resultArr.append(p.apply_async( image_split, args=( captcha_denoising_path, cur_file_list ) ))
    # p.close()
    # p.join()
    print( len(file_list) )
    for i in resultArr:
        i.get()
    end_time = time.time()
    print('图片二值化完毕')
    print('消耗时间：' + str(end_time - start_time))
    # image_binary( '0ABY.jpg' )


