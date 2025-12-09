import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os.path as osp
from PIL import Image


'''
    Single View DR dataset
'''
class SingleViewDataset(Dataset):
    def __init__(self, dataset_name, txt_path, transform=None, select_cls=-1):
        super(SingleViewDataset, self).__init__()

        if dataset_name == 'MESSIDOR':
            self.data_path = './xxx/MESSIDOR/IMAGES/'
            # please replace with your own dataset path
        else:
            self.data_path = './xxx/'
            # please replace with your own dataset path
        self.dataset_name = dataset_name
        self.select_cls = select_cls
        self.txt_path = txt_path
        self.transform = transform
        self.data = []
        self.label = []
        items = self._read_txt(txt_path)
        self._get_data(items)

    def _read_txt(self, txt_file):
        items = []
        split_file = txt_file

        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                if self.dataset_name == 'MESSIDOR':
                    im_all_path, label = line.split(" ")
                    impath = im_all_path.split('/')[-1]
                    impath = osp.join(self.data_path, impath)
                else:
                    impath, label = line.split(" ")
                    impath = osp.join(self.data_path, impath)
                label = int(label)

                if self.select_cls == -1:
                    items.append((impath, label))
                else:
                    if label == self.select_cls: # select DR class
                        items.append((impath, label))
        return items

    def _get_data(self, items):
        impath_label_list = items
        for impath, label in impath_label_list:
            self.data.append(impath)
            self.label.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]
        image = cv2.imread(img_path)

        if image is None:
            print("image error ", img_path, " is not exist!")
            raise ValueError("image error ", img_path, " is not exist!")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        class_id = self.label[index]
        view_id = 0
            
        return image, class_id, view_id 


'''
    Multi View DR dataset
'''
class MultiViewDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None, select_cls=-1, select_view=-1):
        super(MultiViewDataset, self).__init__()

        csv_read_df = pd.read_csv(csv_path)

        if select_cls == -1:
            self.csv_df = csv_read_df
        else:
            self.csv_df = csv_read_df[csv_read_df['level'] == select_cls]
        
        if select_view == -1:
            self.csv_df = self.csv_df
        elif select_view == 0: # get view 0
            temp_csv = self.csv_df
            self.csv_df = temp_csv[temp_csv['id'].str.endswith(('01', '05'))]
        elif select_view == 1: # get view 1
            temp_csv = self.csv_df
            self.csv_df = temp_csv[temp_csv['id'].str.endswith(('02', '06'))]
        elif select_view == 2: # get view 2
            temp_csv = self.csv_df
            self.csv_df = temp_csv[temp_csv['id'].str.endswith(('03', '07'))]
        elif select_view == 3: # get view 3
            temp_csv = self.csv_df
            self.csv_df = temp_csv[temp_csv['id'].str.endswith(('04', '08'))]
        else:
            self.csv_df = self.csv_df
        
        self.img_dir = data_path
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        class_id = self.csv_df.iloc[idx, 1].astype(np.int64)

        img_name = '{}.jpg'.format(self.csv_df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            print("image error ", img_path, " is not exist!")
            raise ValueError("image error ", img_path, " is not exist!")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        view_num = int(self.csv_df.iloc[idx, 0].split('_')[-1])

        if view_num <= 4:
            view_id = view_num -1 
        else:
            view_id = view_num % 5

        return image, class_id, view_id