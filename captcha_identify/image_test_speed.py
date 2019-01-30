from multiprocessing import Pool
import os, numpy as np, time
import asyncio

captcha_path = 'data/captcha'
captcha_test = 'data/captcha_test'

def copy_file( src_path, dest_path, file_list ):
    tasks = []
    loop = asyncio.get_event_loop()
    for cur_file in file_list:
        src_file_name = src_path + '/' + cur_file
        dest_file_name = dest_path + '/' + cur_file
        tasks.append( copy_file_detail(src_file_name, dest_file_name) )

    loop.run_until_complete( asyncio.wait(tasks) )



async def copy_file_detail(src_file_name, dest_file_name):
    with open(src_file_name, 'rb') as fr:
        with open(dest_file_name, 'wb') as fw:
            fw.write(fr.read())

if __name__ == '__main__':
    start_time = time.time()
    p = Pool()
    file_list =  np.array(os.listdir(captcha_path))
    file_list_10 = np.split(file_list, 4, axis=0)
    for cur_file_list in file_list_10:
        # copy_file( captcha_path, captcha_test,cur_file_list )
        p.apply_async(func=copy_file, args=(captcha_path, captcha_test,cur_file_list))
    p.close()
    p.join()
    end_time = time.time()
    print('消耗时间')
    print( end_time - start_time )