from PIL import Image
from sklearn import preprocessing
import numpy as np
import pandas
from pandas import DataFrame
import os, time, asyncio
from multiprocessing import Pool, Queue
import logging,sys
from image_model import trainModel

logger = logging.getLogger('vinkon')
formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(lineno)s][%(levelname)s][%(message)s]')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('log/log.txt')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


captcha_split_type_path = 'data/captcha_split_type'
img_rows = 26
img_cols = 17
model_train_len = 1000
insert_col_value = np.array( [255] * img_rows )


def image_feature( path, child_path, feature_len = model_train_len):

    loop = asyncio.get_event_loop()
    tasks = []
    file_path = os.path.join(path, str(child_path))
    file_list = os.listdir( file_path )[:feature_len]
    for cur_file in file_list:
        # image_denoising_detail(path, cur_file)
        tasks.append( image_feature_detail(file_path, cur_file) )
    result_value = loop.run_until_complete( asyncio.wait( tasks ) )
    final_value = []
    for cur_value in result_value[0]:
        final_value.append( cur_value._result[0] )
    return final_value, child_path


async def image_feature_detail( path, cur_file ):

    img = Image.open(path + '/' + cur_file).convert('L')
    binary_data = np.array(preprocessing.binarize(img, threshold=140))

    feature = get_image_feature( binary_data )
    return feature,

def get_image_feature( binary_data ):
    rows, cols = binary_data.shape[:2]
    dis = img_cols - cols
    paddding_binary_data = binary_data
    if dis > 0:
        left = int(dis / 2)
        right = dis - left
        paddding_binary_data = np.pad(binary_data, ((0, 0), (left, right)), 'constant', constant_values=(255))
    else:
        while paddding_binary_data.shape[1] != img_cols:
            paddding_binary_data = np.delete(paddding_binary_data, -1, axis=1)
    col_feature = (DataFrame( paddding_binary_data ) == 0).astype(int).sum(axis=1)
    row_feature = (DataFrame( paddding_binary_data ) == 0).astype(int).sum(axis=0)
    return np.array(col_feature.append( row_feature ))

def main():

    p = Pool()
    text_types = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    text_types = 'A'
    progress_result = []
    for cur_text_type in text_types:
        # image_feature(captcha_split_type_path, cur_file_list)
        progress_result.append(p.apply_async(image_feature, args=(captcha_split_type_path, cur_text_type)))
    p.close()
    p.join()

    image_arr, label_arr = [], []
    for cur_result in progress_result:
        image, label = cur_result.get()
        image_arr.extend(image)
        label_arr.extend([label] * model_train_len)
    return np.array(image_arr), np.array(label_arr)



if __name__ == '__main__':
    start_time = time.time()
    image_arr, label_arr = main()
    end_time = time.time()
    print('图片二值化完毕')
    print('消耗时间：' + str(end_time - start_time))
    trainModel( image_arr, label_arr )

