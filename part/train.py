import torch
import numpy as np
import os
import my_model as my_model
import torch.nn as nn
import my_dataset
import torch.optim as optim
import datetime
import wandb
import cv2
import random
import time

BATCH_SIZE = 6 
CUDA = 1
WANDB = 1 
LOAD = 1
#cv2.namedWindow("a", cv2.WINDOW_NORMAL);
#cv2.moveWindow("a", 0,0);
#cv2.setWindowProperty("a", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
#cv2.namedWindow("b", cv2.WINDOW_NORMAL);
#cv2.moveWindow("b", 3840,0);
#cv2.setWindowProperty("b", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

if WANDB:
    wandb.init()
    history_directory = '../store/history/%s'%datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(history_directory)
    #wandb.watch(model)

model = my_model.Model()
optimizer = optim.Adam(model.parameters())
if CUDA:
    model = model.cuda()

if LOAD:
    model.load_state_dict(torch.load('../store/history/2020-01-07-15-18-18/2-0.md'))
    optimizer.load_state_dict(torch.load('../store/history/2020-01-07-15-18-18/2-0.adam'))
model.train()


train_set = my_dataset.MyDataset('train')
test_set = my_dataset.MyDataset('test')
train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE,\
        shuffle = True, num_workers = 5, drop_last =  True) 
test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE,\
        shuffle = True, num_workers = 5, drop_last =  True)
loss_f1 = nn.BCELoss()
loss_f2 = nn.MSELoss()

for epoch in range(20000000):
    print('----- epoch %d -----'%epoch) 
    time0 = time.time()
    for i, (inputs, ground_truth) in enumerate(train_loader):
        if CUDA:
            inputs = inputs.cuda()
            ground_truth = ground_truth.cuda()
        outputs = model(inputs)
        loss1 = loss_f2(ground_truth,outputs) ** 0.5 * 256
        loss = loss1
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        if WANDB:
            wandb.log({'loss1':loss1.item(),
                       })

        if i % 10 == 0:
            output_image = outputs[0].permute([1,2,0]).detach().cpu().numpy()
            ground_image = ground_truth[0].permute([1,2,0]).detach().cpu().numpy()
            inputs_image = inputs[0].permute([1,2,0]).detach().cpu().numpy()
            L = 640  
            D = 320
            bar = inputs_image[L // 2 - D //2 : L //2 + D//2, L // 2 - D //2 : L //2 + D//2]
            bar[:] = output_image[:]
            cv2.imshow('ground',ground_image)
            cv2.imshow('input',inputs_image)
            cv2.imshow('output',output_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            os._exit(0)

        if i % 20 == 0:
            print('%d  %10.5f'%(i,loss))

        if i % 5000 == 0 and WANDB:
            torch.save(optimizer.state_dict(),'%s/%d-%d.adam'%(history_directory,epoch,i))
            torch.save(model.state_dict(),'%s/%d-%d.md'%(history_directory,epoch,i))

