import torch
import os
import torch.nn as nn
import cv2
#640         320        160          80         40            20
cfg = [16,16,'M',32, 32, 'M', 64, 64,'M',128,128,'M',256,256, 'M', 512,512,\
       'U',256,256,'U',128,128,'U',128,128,'U',64,64,'U',32,3]
#      40          80          160         320       640

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = self._make_layers(cfg, batch_norm=True, in_channels = 3)
        self._initialize_weights()

    def forward(self, inputs):
        x = self.features(inputs)
        return x


    def _make_layers(self,cfg,batch_norm,in_channels = 3):
        layers = []
        fcn_deep = 128 
        for v in cfg:
#            if v == 'v':
#                layers += [conv2d]
#                continue
            if v == 'M':
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
                n
