from config import config
import os
import csv
import numpy as np
import argparse


# Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption():
    img_paths = []
    captions = []
    
    arg = config.parser.parse_args()

    f = open(arg.caption_file_path, 'r')
    datas = list(csv.reader(f, delimiter='|'))
    f.close()

    for data in datas[1:]:
        img_paths.append(data[0])
        captions.append(data[2])

    return img_paths, captions


# Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(img_paths, captions):
    dataset = np.concatenate(
        (np.reshape(img_paths, (len(img_paths), 1)), 
        np.reshape(captions, (len(captions), 1))), 
        axis = 1
        )

    l = int(len(dataset)*0.8)
    np.save('.\\datasets\\train_dataset', dataset[0:l])
    np.save('.\\datasets\\val_dataset', dataset[l+1:])

    return '.\\datasets\\train_dataset.npy', '.\\datasets\\val_dataset.npy'


# Req. 3-3	저장된 데이터셋 불러오기
def get_data_file(arg, train_dataset_path, val_dataset_path):
    img_paths = []
    captions = []
    
    if arg == 'train' :
        train_datas = np.load(train_dataset_path)
        for data in train_datas:
            img_paths.append(data[0])
            captions.append(data[1])
        return img_paths, captions
    
    if arg == 'test':
        test_datas =np.load(val_dataset_path)
        for data in test_datas:
            img_paths.append(data[0])
            captions.append(data[1])
        return img_paths, captions


# Req. 3-4	데이터 샘플링
def sampling_data(img_paths, caption):
    random_idx = np.random.randint(len(img_paths))
    return img_paths[random_idx], caption[random_idx]