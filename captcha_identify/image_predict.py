from sklearn.externals import joblib
from image_feature import image_feature, image_feature_detail, get_image_feature
from PIL import Image
import numpy as np
from sklearn import preprocessing

model_path = 'model/test.model'
test_captcha_path = 'data/captcha'
text_num = 4

def image_transfer(path, image_name):
    img = Image.open(path + '/' + image_name).convert('L')
    binary_data = np.array(preprocessing.binarize(img, threshold=140), dtype='int')
    img.close()
    binary_data[binary_data == 1] = 255
    return binary_data

def image_denosing( binary_data ):
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

    return binary_data


def image_split( binary_data ):
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
            split_images.append(binary_data[:, start1:end1])
            split_images.append(binary_data[:, start2:end2])
            continue;
        split_images.append( binary_data[:,start:end])
    return split_images


def main():
    test_file = '0GG8.jpg'
    binary_data = image_transfer( test_captcha_path, test_file )
    binary_data = image_denosing( binary_data )
    split_binary_data = image_split( binary_data )
    get_image_feature( binary_data )
    feature = []
    for cur_binary_data in split_binary_data:
        feature.append( get_image_feature(cur_binary_data) )
    model = joblib.load(model_path)
    print(model.predict(feature))


if __name__ == '__main__':
    main()

