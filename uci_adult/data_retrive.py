#-*- coding: utf-8 -*-

import os
from urllib.request import urlretrieve
import traceback

def reporthook(block_read,block_size,total_size):
  if not block_read:
    print("connection opened")
    return
  if total_size<0:
    #unknown size
    print("read %d blocks (%dbytes)" %(block_read,block_read*block_size))
    # logger.warn('It seems wrong, since total_size cannot get, just skip this')
    # return
  else:
    amount_read=block_read*block_size;
    print ('Read %d blocks,or %d/%d, or %f' %(block_read,amount_read,total_size, (amount_read+0.0)/total_size))
  return

def download_link(directory, pubTitle, link):
    print('Downloading %s', link)
    if(pubTitle == '' or link == ''):
        print('the link or pubTitle is just wrong!')
        return

    download_path = os.path.join(directory, pubTitle)
    # download_path = os.path.join(directory)
    print(os.path.abspath(download_path))
    
    if os.path.isfile(download_path):
        print ('Alreay downloaded', pubTitle)
        return
    try:
        urlretrieve(link,download_path,reporthook)        
        print('Download %s done', link)
    except Exception as e:
        print(e)
        print('Wrong in download')
        return

if __name__ == '__main__':
    baseUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'


    to_download = ['Index', 'adult.data', 'adult.names', 'adult.test', 'old.adult.names']

    for d in to_download:
        download_link('adult_data' ,d, baseUrl+d)

    t_t_l = ('adult-test-features.csv','https://raw.githubusercontent.com/jvpoulos/cs289-project/master/adult-dataset/adult-test-features.csv')
    download_link('adult_data' ,t_t_l[0], t_t_l[1])