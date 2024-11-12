import os
import re
import random
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np

class Tactile_Vision_dataset(data.Dataset):
    def __init__(self, data_path='./data', transform=False, num=8):
        self.data_path = data_path
        self.label_files = []
        self.train_data = []
        for root, dirs, files in os.walk(data_path, topdown=True):
            for file in files:
                if file.endswith('.dat'):
                    self.label_files.append(os.path.join(root, file))
        # for file in os.listdir(data_path):
        #     if file.endswith('.dat'):
        #         self.label_files.append(os.path.join(data_path, file))
        
        self.num = num

        pat = re.compile(r'object([0-9]+)_result')
        for label_file in self.label_files:
            idx = pat.search(label_file).group(1)
            fp = open(label_file, 'r')
            lines = fp.readlines()
            # self.train_data.extend([line.replace('\n','') + ' ' + idx for line in lines])
            
            for line in lines:
                line = line.replace('\n','') + ' ' + idx
                data = line.split(' ')
                object_id = data[-1]
                id_1 = data[-2]
                id_2 = data[-3]
                status = int(data[2][0]) # Label
                label = torch.tensor([status]).long()
                label = torch.squeeze(label)

                path = os.path.join(self.data_path, 'object' + object_id, id_2 + '_mm')
                count = 0
                for root, dirs, files in os.walk(path, topdown=True):
                    for file in files:
                        if "external_" in file and file.endswith(".jpg"):
                            count += 1
                for i in range(count-self.num):
                    self.train_data.append(line+' '+str(i))
                           
              
        self.transform = transform
        if self.transform:
            self.transH = transforms.RandomHorizontalFlip(p=1)
            self.transV = transforms.RandomVerticalFlip(p=1)
            self.trans = transforms.Resize([224,224])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        train_data = self.train_data[index]
        output_tacitle_imgs = []
        output_rgb_imgs = []

        train_data = train_data.split(' ')
        object_id = train_data[-2]  #MTAG
        id_1 = train_data[-3]       #MTAG
        id_2 = train_data[-4]       #MTAG
        status = int(train_data[2][0]) # Label
        label = torch.tensor([status]).long()
        label = torch.squeeze(label)

        path = os.path.join(self.data_path, 'object' + object_id, id_2 + '_mm')
        rgb_img_paths = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                if ("external_" in file) and file.endswith('.jpg'):
                    rgb_img_paths.append(os.path.join(root, file))

        random_num = []
        # MTODO 随机采样
        # while(len(random_num)<8):
        #     x = random.randint(0, len(rgb_img_paths) - 1)
        #     if x not in random_num:
        #         random_num.append(x)
        # random_num.sort()
        # x = random.randint(0,len(rgb_img_paths)-9)
        
        x = int(train_data[-1])
        for i in range(self.num):
            random_num.append(x+i)
        random_num.sort()

        if self.transform:
            rgb_img_transH, tacitle_img_transH, rgb_img_transV, tacitle_img_transV = False, False, False, False
            if random.random() < 0.5:
                rgb_img_transV = True
            if random.random() < 0.5:
                tacitle_img_transV = True
            
            if random.random() < 0.5:
                rgb_img_transH = True
            if random.random() < 0.5:
                tacitle_img_transH = True

        for i in random_num:
            rgb_img_path = rgb_img_paths[i]
            cor_tacitle_img_path = rgb_img_path.replace('external', 'gelsight')
            rgb_img = cv2.imread(rgb_img_path)
            tacitle_img = cv2.imread(cor_tacitle_img_path)
                
            if self.transform:
                rgb_img = self.trans(Image.fromarray(rgb_img))
                tacitle_img = self.trans(Image.fromarray(tacitle_img))
                if rgb_img_transH:
                    rgb_img = self.transH(rgb_img)
                if tacitle_img_transH:
                    tacitle_img = self.transH(tacitle_img)
                
                if rgb_img_transV:
                    rgb_img = self.transV(rgb_img)
                if tacitle_img_transV:
                    tacitle_img = self.transV(tacitle_img)
                    
                rgb_img = np.array(rgb_img)
                tacitle_img = np.array(tacitle_img)
                
            size = rgb_img.shape
                        
            rgb_img_tensor = torch.from_numpy(rgb_img.reshape(size[2], size[0], size[1])).float()
            tacitle_img_tensor = torch.from_numpy(tacitle_img.reshape(size[2], size[0], size[1])).float()
            output_rgb_imgs.append(rgb_img_tensor)
            output_tacitle_imgs.append(tacitle_img_tensor)

        return output_rgb_imgs, output_tacitle_imgs, label

