import torch
import os
import torch.nn as nn
import cv2

#     160          80         40            20          10
cfg_gen = [32, 32, 'M', 64, 64,'M',128,128,'M',256,256, 'M', 256,'k',\
       'U',256,256,'U',128,128,'U',64,64,'v']
#       20          40          80                    
#
#     160          80         40          20             10
cfg_judge_i = [32, 32, 'M', 64, 64,'M',128,128,'M',256,256, 'M', 512,512]
cfg_judge_o = [64, 64,'M',128,128,'M',256,256, 'M', 512,512]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gen_cnn = self._make_layers(cfg_gen, batch_norm=True, in_channels = 3)
        self.judge_cnn_i = self._make_layers(cfg_judge_i, batch_norm=True, in_channels=3)
        self.judge_cnn_o = self._make_layers(cfg_judge_o, batch_norm=True, in_channels=3)
        self.judge_score = nn.Sequential(
            #nn.Linear(10*10*512*2, 256),
            nn.Linear(10*10*512, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
        self._initialize_weights()

    def forward(self, inputs):
        pass

    def open_grad(self,gan):
        if gan == 'gen':
            a = [True,False,False]
        else:
            a = [False,True,True]
        for param in self.gen_cnn.parameters():
            param.requires_grad = a[0]
        for param in self.judge_cnn_i.parameters():
            param.requires_grad = a[1]
        for param in self.judge_cnn_o.parameters():
            param.requires_grad = a[1]
        for param in self.judge_score.parameters():
            param.requires_grad = a[2]
        

    def forward(self, inputs, groud_truth,grad='gen'):
        self.open_grad(grad)
        gen_image = self.gen_cnn(inputs)
        x = self.judge_cnn_i(inputs)
        x = x.flatten(1) 
        y = self.judge_cnn_o(gen_image)
        y = y.flatten(1) 
        #z = torch.cat([x,y],1)
        z = (x - y) ** 2
        conf_for_gen = self.judge_score(z)
        x = self.judge_cnn_i(inputs)
        x = x.flatten(1) 
        y = self.judge_cnn_o(groud_truth)
        y = y.flatten(1) 
        #z = torch.cat([x,y],1)
        z = (x - y) ** 2
        conf_for_real = self.judge_score(z)
        return gen_image, conf_for_gen, conf_for_real

    def _make_layers(self,cfg,batch_norm,in_channels = 3):
        layers = []
        fcn_deep = 128 
        for v in cfg:
            if v == 'v':
                conv2d = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(3)]
                #conv2d = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
                #layers += [conv2d, nn.ReLU()]
                #layers += [conv2d]
                continue
            if v == 'k':
                conv2d = nn.Conv2d(256, 256, kernel_size=5, padding=2, bias=False)
                layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU()]
                continue
            if v == 'M':
                #layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            if v == 'U':
                layers += [nn.Upsample(scale_factor=2,mode = 'nearest')]
                continue
            if 1:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for i,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                   nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
