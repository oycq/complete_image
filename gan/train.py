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

BATCH_SIZE = 5 
CUDA = 1
WANDB = 1
LOAD = 0
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
optimizer_D = optim.Adam(model.parameters())
optimizer_G = optim.Adam(model.parameters())
if CUDA:
    model = model.cuda()

if LOAD:
    model.load_state_dict(torch.load('../store/history/2020-01-08-00-39-51/25-5000.md'))
    optimizer.load_state_dict(torch.load('../store/history/2020-01-08-00-39-51/25-5000.adam'))
model.train()


train_set = my_dataset.MyDataset('train')
test_set = my_dataset.MyDataset('test')
train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE,\
        shuffle = True, num_workers = 5, drop_last =  True) 
test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE,\
        shuffle = True, num_workers = 5, drop_last =  True)
loss_f1 = nn.BCELoss()
loss_f2 = nn.MSELoss()
gd = torch.zeros((BATCH_SIZE,1))
if CUDA:
    gd = gd.cuda()

first_time = 1
if_G = 1
count = 0
stable_count = 10
for epoch in range(20000000):
    print('----- epoch %d -----'%epoch) 
    time0 = time.time()
    for i, (inputs, ground_truth) in enumerate(train_loader):
        if CUDA:
            inputs = inputs.cuda()
            ground_truth = ground_truth.cuda()

        
        if if_G == 1: 
            gen_image, conf_for_gen = model.gen(inputs)
            loss1 = loss_f2(gd+1,conf_for_gen) 
            optimizer_G.zero_grad() 
            loss1.backward()
            optimizer_G.step()
            if loss1 < 0.36:
                count += 1
                if count > stable_count:
                    if_G = 0
                    count = 0
            else:
                count = 0

        if if_G == 0 or first_time == 1:
            first_time = 0
            conf_for_fake,conf_for_real,fake_image = model.judge(ground_truth,inputs)
            loss2 = loss_f2(conf_for_fake,gd) / 2 + loss_f2(conf_for_real,gd+1) / 2
            optimizer_D.zero_grad() 
            loss2.backward()
            optimizer_D.step()
            if loss2 < 0.16:
                count += 1
                if count > stable_count:
                    count = 0
                    if_G = 1
            else:
                if_G = 0
                count = 0


        if WANDB:
            wandb.log({'conf_for_gen':torch.mean(conf_for_gen).item(),
                        'conf_for_fake':torch.mean(conf_for_fake).item(),
                        'conf_for_real':torch.mean(conf_for_real).item(),
                        'loss1':loss1.item(),
                        'loss2':loss2.item(),
                        'if_G':if_G,
                       })

        if i % 10 == 0:
            if if_G == 1:
                output_image = gen_image[0].permute([1,2,0]).detach().cpu().numpy()
            else:
                output_image = fake_image[0].permute([1,2,0]).detach().cpu().numpy()
            ground_image = ground_truth[0].permute([1,2,0]).detach().cpu().numpy()
            cv2.imshow('ground',ground_image)
            cv2.imshow('input',output_image)
            if WANDB and i % 100 == 0:
                output_image = (output_image * 255).astype('uint8')
                ground_image= (ground_image* 255).astype('uint8')
                cv2.imwrite('%s/gen-%d-%d.jpg'%(history_directory,epoch,i),output_image)
                cv2.imwrite('%s/real-%d-%d.jpg'%(history_directory,epoch,i),ground_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            os._exit(0)

        if i % 20 == 0:
            print('%5d  G:%2d gen: %10.5f real_conf:%10.5f fake_conf:%10.5f'%(i,if_G,\
                    torch.mean(conf_for_gen).item(),\
                torch.mean(conf_for_real).item(),torch.mean(conf_for_fake).item()
                    ))

    if epoch % 2 == 0 and WANDB:
        torch.save(optimizer_G.state_dict(),'%s/%d-%d.gadam'%(history_directory,epoch,i))
        torch.save(optimizer_D.state_dict(),'%s/%d-%d.dadam'%(history_directory,epoch,i))
        torch.save(model.state_dict(),'%s/%d-%d.md'%(history_directory,epoch,i))

