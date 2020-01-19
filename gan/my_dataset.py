import cv2
import glob
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils import data


L = 640
D = 320

def get_object_box_color(obeject_id):
    b = (obeject_id // 9 - 5) * 30 + 150
    g = (obeject_id % 10 - 5) * 30 + 150
    r = (obeject_id % 9 - 5) * 30 + 150
    return (b,g,r)


label_file_name_list = sorted(glob.glob('../../labels/train2014/*.txt'))

#def get_data(index):
#    label_file_name = label_file_name_list[index]
#    image_file_name = label_file_name.replace('labels','images').replace('txt','jpg')
#    image = cv2.imread(image_file_name)
#    label_file = open(label_file_name)
#    label_txt_lines = label_file.readlines()
#    W,H = image.shape[1],image.shape[0]
#    dw1 = (L-W)//2
#    dw2 = L - W  - dw1
#    dh1 = (L-H)//2
#    dh2 = L - H - dh1
#    image = cv2.copyMakeBorder(image, dh1,dh2,dw1,dw2, cv2.BORDER_CONSTANT,value=(0,0,0))
#    image_raw = image.copy()
#    bar = image[L // 2 - D //2 : L //2 + D//2, L // 2 - D //2 : L //2 + D//2]
#    roi = bar.copy()
#    bar *= 0
##    for txt_line in label_txt_lines:
##        txt_line = txt_line[:-2]
##        obeject_id, x,y,w,h = txt_line.split(' ')
##        obeject_id = int(obeject_id)
##        x,y,w,h = float(x),float(y),float(w),float(h)
##        x,y,w,h = int(x*W-w/2*W),int(y*H-h/2*H),int(w*W),int(h*H)
##        x,y = x + dw1, y + dh1
##        box_color = get_object_box_color(random.randrange(0,81))
##        cv2.rectangle(image, (x,y), (x+w,y+h),box_color,2)
##    cv2.imshow('image',image)
##    cv2.imshow('roi',cv2.resize(roi, (640,640),interpolation=cv2.INTER_NEAREST))
#    return image,image_raw, roi

def get_data(index):
    label_file_name = label_file_name_list[index]
    image_file_name = label_file_name.replace('labels','images').replace('txt','jpg')
    image = cv2.imread(image_file_name)
    label_file = open(label_file_name)
    label_txt_lines = label_file.readlines()
    W,H = image.shape[1],image.shape[0]
    dw1 = (L-W)//2
    dw2 = L - W  - dw1
    dh1 = (L-H)//2
    dh2 = L - H - dh1
    image = cv2.copyMakeBorder(image, dh1,dh2,dw1,dw2, cv2.BORDER_CONSTANT,value=(0,0,0))
    image = image[L // 2 - D //4 : L //2 + D//4, L // 2 - D //4 : L //2 + D//4]
    image_raw = image.copy()
    bar = image[L // 8 - D //8 : L //8 + D//8, L // 8 - D //8 : L //8 + D//8]
    roi = bar.copy()
    bar *= 0
    return image,image_raw, roi

#fake_index  = 0

class MyDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test

    def __len__(self):
        if self.train_test ==  'train':
            return len(label_file_name_list)* 2 // 3
        else:
            return len(label_file_name_list)// 3 - 1

    def __getitem__(self,index):
#        global fake_index 
#        fake_index = (fake_index + 1) % 10
        if self.train_test == 'test':
            index += len(points_list) * 2 // 3
#        image, image_raw, roi =  get_data(fake_index)
        image, image_raw, roi =  get_data(index)
        inputs = torch.tensor(image,dtype = torch.float32) / 255
        inputs = inputs.permute([2,0,1])
        ground_truth = torch.tensor(image_raw,dtype = torch.float32) / 255
        ground_truth = ground_truth.permute([2,0,1])
        roi = torch.tensor(roi,dtype = torch.float32) / 255
        roi = roi.permute([2,0,1])
        return inputs, roi 

if __name__ == '__main__':
    for i in range(len(label_file_name_list)):
        image,image_raw,roi = get_data(i)
        cv2.imshow('image',image)
        cv2.imshow('image_raw',image_raw)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

