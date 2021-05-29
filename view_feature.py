import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import math
from torch.nn import init
from torch.nn import functional as F
from IPython import embed

savepath='../features_personone3'
if not os.path.exists(savepath):
    os.mkdir(savepath)


def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        embed()
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


class ft_net(nn.Module):

    def __init__(self,block, layers):
        super(ft_net, self).__init__()
        self.inplanes = 64
        #self.loss = loss
        #self.aligned = aligned

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8,1))
        #self.classifier = nn.Linear(2048,num_classes)
        #self.horizon_pool = HorizontalMaxPool2D()
        #if self.aligned:
        #    self.bn = nn.BatchNorm2d(2048)
        #    self.relu1 = nn.ReLU(inplace=True)
        #    self.conv2 = nn.Conv2d(2048,128,kernel_size=1,stride=1,padding=0,bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if True: # draw features or not
            x = self.conv1(x)
            #draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))

            x = self.bn1(x)
            #draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))

            x = self.relu(x)
            #draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))

            x = self.maxpool(x)
            #draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.layer1(x)
            #draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

            x = self.layer2(x)
            #draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

            x = self.layer3(x)
            draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

            x = self.layer4(x)
            draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))

            x = self.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

           
        else :
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
        

        return x
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

pretrained_model = models.resnet50(pretrained=True)
model = ft_net(block = Bottleneck, layers = [3, 4, 6, 3])
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# pretrained_dict = resnet50.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
model=model.cuda()
model.eval()
img=cv2.imread('/data-tmp/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg')
img=cv2.resize(img,(128,256));
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485,0.48556,0.406],std=[0.229,0.22924,0.225])])
img=transform(img).cuda()
img=img.unsqueeze(0)
with torch.no_grad():
    start=time.time()
    out=model(img)
    print("total time:{}".format(time.time()-start))
    result=out.cpu().numpy()
    # ind=np.argmax(out.cpu().numpy())
    ind=np.argsort(result,axis=1)
    for i in range(5):
        print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
    print("done")
